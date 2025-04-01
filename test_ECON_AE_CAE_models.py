import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf

# Configure GPUs (if available)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

# Keras and QKeras
from tensorflow import keras
from keras.layers import (Layer, Input, Flatten, Dense, ReLU, Reshape, 
                          Conv2DTranspose, concatenate)
from keras.models import Model
import qkeras
from qkeras import QActivation, QConv2D, QDense, quantized_bits

# Plotting
import matplotlib.pyplot as plt

# Custom utilities (assumes you have these in telescope.py and utils.py)
from utils.telescope import telescopeMSE8x8
from utils.utils import (ArgumentParser, load_pre_processed_data, mean_mse_loss, 
                         cos_warm_restarts, cosine_annealing, save_models)


################################################################
# Custom Keras Layers (unchanged)
################################################################
class keras_pad(Layer):
    def call(self, x):
        padding = tf.constant([[0, 0], [0, 1], [0, 1], [0, 0]])
        return tf.pad(x, padding, mode='CONSTANT', constant_values=0)

class keras_minimum(Layer):
    def call(self, x, sat_val=1):
        return tf.minimum(x, sat_val)

class keras_floor(Layer):
    def call(self, x):
        if isinstance(x, tf.SparseTensor):
            x = tf.sparse.to_dense(x)
        return tf.math.floor(x)

################################################################
# Parse Command-Line Arguments
################################################################
p = ArgumentParser()

# Paths
p.add_argument('--mpath', type=str, required=False)
p.add_argument('--m', type=int, required=False)

# Dataset parameters
p.add_argument('--data_path', type=str, required=True)
p.add_argument('--num_files', type=int, required=True)
p.add_argument('--train_dataset_size', type=int, default=500000)
p.add_argument('--val_dataset_size', type=int, default=100000)
p.add_argument('--test_dataset_size', type=int, default=100000)

args = p.parse_args()

################################################################
# Determine Model(s) to Validate
################################################################
all_models = [2, 3, 4, 5]
bitsPerOutputLink = [0,  1,  3,  5,  7,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9]    

################################################################
# Main Loop Over the eLink or Bit Configurations
################################################################
m = args.m

################################################################
# Load Data
################################################################
batch = 1

output_dir = os.path.join(args.mpath)
eLinks = m
bitsPerOutput = bitsPerOutputLink[eLinks]
print(f"Validating Model with {eLinks} eLinks")
model_dir = os.path.join(output_dir, f"model_{eLinks}_eLinks")

################################################################
# Model Hyperparameters
################################################################
nIntegerBits = 1
nDecimalBits = bitsPerOutput - nIntegerBits
outputSaturationValue = (1 << nIntegerBits) - 1./(1 << nDecimalBits)
maxBitsPerOutput = 9
outputMaxIntSize = 1 if (bitsPerOutput <= 0) else (1 << nDecimalBits)
outputMaxIntSizeGlobal = 1 if (maxBitsPerOutput <= 0) else (1 << (maxBitsPerOutput - nIntegerBits))

batch = args.batchsize
n_kernels = 8
n_encoded = 16
conv_weightBits = 6
conv_biasBits   = 6
dense_weightBits = 6
dense_biasBits   = 6
encodedBits = 9
CNN_kernel_size = 3

################################################################
# Encoder Definition
################################################################

# Encoder Inputs
input_enc = Input(batch_shape=(batch, 8, 8, 1), name='Wafer')
cond      = Input(batch_shape=(batch, 8), name='Cond')

# Quantize input (8-bit quant, 1 integer bit)
x = QActivation(quantized_bits(bits=8, integer=1), name='input_quantization')(input_enc)

# Zero-pad so the next layer can stride properly
x = keras_pad()(x)

# Convolution
x = QConv2D(
    n_kernels,
    CNN_kernel_size, 
    strides=2, 
    padding='valid',
    kernel_quantizer=quantized_bits(bits=conv_weightBits, integer=0, keep_negative=1, alpha=1),
    bias_quantizer=quantized_bits(bits=conv_biasBits,   integer=0, keep_negative=1, alpha=1),
    name="conv2d"
)(x)

# Activation (8-bit quant)
x = QActivation(
    quantized_bits(bits=8, integer=1), 
    name='act'
)(x)

# Flatten for Dense
x = Flatten()(x)

# Dense layer
x = QDense(
    n_encoded,
    kernel_quantizer=quantized_bits(bits=dense_weightBits, integer=0, keep_negative=1, alpha=1),
    bias_quantizer=quantized_bits(bits=dense_biasBits,     integer=0, keep_negative=1, alpha=1),
    name="dense"
)(x)

# Quantize latent space (9-bit quant, 1 integer bit)
latent = QActivation(
    qkeras.quantized_bits(bits=encodedBits, integer=nIntegerBits),
    name='latent_quantization'
)(x)

# If bits are allocated for output, rescale and saturate
if bitsPerOutput > 0 and maxBitsPerOutput > 0:
    latent = keras_floor()(latent * outputMaxIntSize)
    latent = keras_minimum()(latent / outputMaxIntSize, sat_val=outputSaturationValue)

# Concatenate conditions
latent = concatenate([latent, cond], axis=1)

# Build the encoder model
encoder = keras.Model([input_enc, cond], latent, name="encoder")

################################################################
# Decoder Definition
################################################################

# Decoder input
input_dec = Input(batch_shape=(batch, 24))

# Simple multi-layer perceptron
y = Dense(24)(input_dec)
y = ReLU()(y)
y = Dense(64)(y)
y = ReLU()(y)
y = Dense(128)(y)
y = ReLU()(y)

# Reshape to feature map
y = Reshape((4, 4, 8))(y)

# Deconvolution (Conv2DTranspose)
y = Conv2DTranspose(1, (3, 3), strides=(2, 2), padding='valid')(y)

# Slice to 8x8
y = y[:, 0:8, 0:8]
y = ReLU()(y)

recon = y

# Build the decoder model
decoder = keras.Model([input_dec], recon, name="decoder")

################################################################
# Full Autoencoder (Encoder + Decoder)
################################################################

cae = Model(
    inputs=[input_enc, cond],
    outputs=decoder([encoder([input_enc, cond])]),
    name="cae"
)

cae_path = os.path.join(model_dir, 'final_best_model.hdf5')

cae.load_weights(cae_path)


print('Loading Data...')
train_loader, test_loader, val_loader = load_pre_processed_data(args.num_files, batch, m, args)
print('Data Loaded!')

################################################################
# Inference and Validation using the Frozen Graphs
################################################################
total_loss_val = 0.0
num_batches = 0
all_output = []
all_inputs = [] 
i = 0
for wafers, cond_data in test_loader:
    # Run encoder inference.
    output = cae.predict([wafers, cond_data])
    all_output.append(output)
    all_inputs.append(wafers)
    if i == 200000:
        break
    i += 1
    num_batches += 1
    
all_inputs = np.concatenate(all_inputs, axis=0)
all_output = np.concatenate(all_output, axis=0)
total_loss = telescopeMSE8x8(all_inputs, all_output)
total_loss_val += total_loss
if num_batches > 0:
    total_loss_val /= num_batches


print(f"Test Loss: {float(total_loss_val):.8f}")

