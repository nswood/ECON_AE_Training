import os
import pickle
import tensorflow as tf
import sys
import time
import math
import yaml
import inspect
import numpy as np
import pandas as pd
from argparse import SUPPRESS, ArgumentParser as _AP
import keras.backend as K
from qkeras.utils import model_save_quantized_weights
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from uuid import uuid4
from datetime import datetime

# Keras / QKeras imports
from tensorflow import keras
from keras.layers import (Input, Flatten, Dense, ReLU, Reshape, Conv2DTranspose)
from keras.models import Model
from qkeras import QActivation, QConv2D, QDense, quantized_bits

# Custom modules
from utils.telescope import telescopeMSE8x8  # If you need to reference the loss
from utils.utils import ArgumentParser, save_CMSSW_models
import utils.graph as graph  # For writing frozen graphs
def save_CMSSW_compatible_model(mname, mpath, outdir, eLinks, alloc_geom):
    ##############################################################################
    # Global Configuration
    ##############################################################################
    
    # Destination directory for the CMSSW-friendly models
    loading_dir = mpath
    saving_dir = outdir
    out_subdir = outdir
    if not os.path.exists(loading_dir):
        os.system("mkdir -p " + loading_dir)
    if not os.path.exists(saving_dir):
        os.system("mkdir -p " + saving_dir)

    # If you map eLinks to bits, set them up here
    bitsPerOutputLink = [
        0, 1, 3, 5, 7, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9
    ]
    
    bitsPerOutput = bitsPerOutputLink[eLinks]
    print(f"Preparing Model with {eLinks} eLinks")
    # out_subdir = os.path.join(saving_dir, f"hyperband_search_{eLinks}", f"best_model_{eLinks}_eLinks_for_CMSSW")
    
    
    if not os.path.exists(out_subdir):
        os.system("mkdir -p " + out_subdir)
    
    # Setup fixed parameters
    batch = 1
    nIntegerBits = 1
    nDecimalBits = bitsPerOutput - nIntegerBits
    outputSaturationValue = (1 << nIntegerBits) - 1. / (1 << nDecimalBits) if bitsPerOutput > 0 else 1
    maxBitsPerOutput = 9
    
    outputMaxIntSize = 1
    if bitsPerOutput > 0:
        outputMaxIntSize = (1 << nDecimalBits)
    
    outputMaxIntSizeGlobal = 1
    if maxBitsPerOutput > 0:
        outputMaxIntSizeGlobal = (1 << (maxBitsPerOutput - nIntegerBits))
    
    # Model layout hyperparameters
    n_kernels = 8
    n_encoded = 16
    conv_weightBits = 6
    conv_biasBits   = 6
    dense_weightBits = 6
    dense_biasBits   = 6
    encodedBits = 9
    CNN_kernel_size = 3
    padding = tf.constant([[0, 0], [0, 1], [0, 1], [0, 0]])
    
    ##############################################################################
    # Build Encoder
    ##############################################################################
    
    input_enc = keras.Input(batch_shape=(batch, 8, 8, 1), name='Wafer')
    
    # Quantize input: 8 bits, 1 integer bit
    x = QActivation(quantized_bits(bits=8, integer=1), name='input_quantization')(input_enc)
    
    # Pad to (9x9)
    x = tf.pad(x, padding, mode='CONSTANT', constant_values=0, name=None)
    
    # Convolution
    x = QConv2D(
        n_kernels,
        CNN_kernel_size,
        strides=2,
        padding='valid',
        kernel_quantizer=quantized_bits(bits=conv_weightBits, integer=0, keep_negative=1, alpha=1),
        bias_quantizer=quantized_bits(bits=conv_biasBits,     integer=0, keep_negative=1, alpha=1),
        name="conv2d"
    )(x)
    
    # Activation
    x = QActivation(
        quantized_bits(bits=8, integer=1), 
        name='act'
    )(x)
    
    # Flatten and Dense
    x = Flatten()(x)
    x = QDense(
        n_encoded,
        kernel_quantizer=quantized_bits(bits=dense_weightBits, integer=0, keep_negative=1, alpha=1),
        bias_quantizer=quantized_bits(bits=dense_biasBits,     integer=0, keep_negative=1, alpha=1),
        name="dense"
    )(x)
    
    # Quantize latent (9 bits, 1 integer bit)
    x = QActivation(
        quantized_bits(bits=encodedBits, integer=1),
        name='latent_quantization'
    )(x)
    
    # Apply floor+min for bit saturation if needed
    if bitsPerOutput > 0 and maxBitsPerOutput > 0:
        x_floor = tf.math.floor(x * outputMaxIntSize)
        x = tf.minimum(x_floor / outputMaxIntSize, outputSaturationValue)
    
    encoder = keras.Model([input_enc], x, name="encoder")
    
    ##############################################################################
    # Build Decoder
    ##############################################################################
    
    input_dec = keras.Input(batch_shape=(batch, 24))
    y = Dense(24)(input_dec)
    y = ReLU()(y)
    y = Dense(64)(y)
    y = ReLU()(y)
    y = Dense(128)(y)
    y = ReLU()(y)
    y = Reshape((4, 4, 8))(y)
    
    y = Conv2DTranspose(1, (3, 3), strides=(2, 2), padding='valid')(y)
    # Slice to (8x8)
    y = y[:, 0:8, 0:8]
    y = ReLU()(y)
    
    decoder = keras.Model([input_dec], y, name="decoder")
    
    ##############################################################################
    # Load Weights
    ##############################################################################
    
    print(os.listdir(loading_dir))
    
    encoder_path = os.path.join(loading_dir, "encoder_model.hdf5")
    decoder_path = os.path.join(loading_dir, "decoder_model.hdf5")
    
    encoder.load_weights(encoder_path)
    decoder.load_weights(decoder_path)
    
    # Compile dummy model to finalize graph (optional for QKeras usage)
    loss = telescopeMSE8x8  # if you need the same telescope loss
    opt = tf.keras.optimizers.Adam(learning_rate=0.1, weight_decay=0.000025)
    encoder.compile(optimizer=opt, loss=loss)
    decoder.compile(optimizer=opt, loss=loss)
    
    ##############################################################################
    # Save Frozen Graphs for CMSSW
    ##############################################################################
    
    save_CMSSW_models(encoder, decoder, out_subdir, mname, isQK=True)
