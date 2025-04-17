import os
import sys
import time
import math
import yaml
import pickle
import inspect
import numpy as np
import pandas as pd
import tensorflow as tf
from argparse import SUPPRESS, ArgumentParser as _AP
import keras.backend as K
from qkeras.utils import model_save_quantized_weights
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
import utils.graph as graph
from uuid import uuid4
from datetime import datetime
from tensorflow import keras
from keras.layers import (Layer, Input, Flatten, Dense, ReLU, Reshape,
                          Conv2DTranspose, concatenate)
from keras.models import Model
import qkeras
from qkeras import QActivation, QConv2D, QDense, quantized_bits


from utils.custom_layers import *
from utils.regularization_layers.regularizers import *

##############################################################################
# Model Building
##############################################################################

def build_cae_model(bitsPerOutput, orthogonal_regularization_factor = 0):
    if orthogonal_regularization_factor > 0:
        
        # n_encoded (just an example of an architecture hyperparam you can tune)
        n_encoded = 16

        # -----------------------------------------------------------
        # 3) Bits logic from your code
        # -----------------------------------------------------------
        nIntegerBits = 1
        nDecimalBits = bitsPerOutput - nIntegerBits
        outputSaturationValue = (1 << nIntegerBits) - 1./(1 << nDecimalBits)
        maxBitsPerOutput = 9
        outputMaxIntSize = 1 if (bitsPerOutput <= 0) else (1 << nDecimalBits)

        # Additional fixed settings
        n_kernels = 8
        conv_weightBits = 6
        conv_biasBits   = 6
        dense_weightBits = 6
        dense_biasBits   = 6
        encodedBits = 9
        CNN_kernel_size = 3

        regularizer = tf.keras.regularizers.OrthogonalRegularizer(
            factor=orthogonal_regularization_factor, mode='rows'
        )

        de_conv_regularizer = DeconvOrthRegularizer(stride=2, padding='VALID', coeff=orthogonal_regularization_factor)

        # -----------------------------------------------------------
        # 4) Build Encoder
        # -----------------------------------------------------------
        input_enc = Input(shape=(8, 8, 1), name='Wafer')
        cond      = Input(shape=(8,), name='Cond')

        x = QActivation(quantized_bits(bits=8, integer=1), name='input_quantization')(input_enc)
        x = keras_pad()(x)
        x = QConv2D(
            n_kernels,
            CNN_kernel_size, 
            strides=2, 
            padding='valid',
            kernel_quantizer=quantized_bits(bits=conv_weightBits, integer=0, keep_negative=1, alpha=1),
            bias_quantizer=quantized_bits(bits=conv_biasBits,   integer=0, keep_negative=1, alpha=1),
            name="conv2d",
            kernel_regularizer=de_conv_regularizer
        )(x)
        x = QActivation(quantized_bits(bits=8, integer=1), name='act')(x)
        x = Flatten()(x)
        x = QDense(
            n_encoded,
            kernel_quantizer=quantized_bits(bits=dense_weightBits, integer=0, keep_negative=1, alpha=1),
            bias_quantizer=quantized_bits(bits=dense_biasBits,     integer=0, keep_negative=1, alpha=1),
            name="dense",
            kernel_regularizer=regularizer
        )(x)

        latent = QActivation(
            qkeras.quantized_bits(bits=encodedBits, integer=nIntegerBits),
            name='latent_quantization'
        )(x)

        # If bits are allocated, rescale + saturate
        if bitsPerOutput > 0 and maxBitsPerOutput > 0:
            latent = keras_floor()(latent * outputMaxIntSize)
            latent = keras_minimum()(latent / outputMaxIntSize, sat_val=outputSaturationValue)

        # Concatenate condition
        latent = concatenate([latent, cond], axis=1)
        encoder = keras.Model([input_enc, cond], latent, name="encoder")

        # -----------------------------------------------------------
        # 5) Build Decoder
        # -----------------------------------------------------------
        input_dec = Input(shape=(24,))
        y = Dense(24,kernel_regularizer=regularizer)(input_dec)
        y = ReLU()(y)
        y = Dense(64,kernel_regularizer=regularizer)(y)
        y = ReLU()(y)
        y = Dense(128,kernel_regularizer=regularizer)(y)
        y = ReLU()(y)
        y = Reshape((4, 4, 8))(y)
        y = Conv2DTranspose(1, (3,3), strides=(2,2), padding='valid',kernel_regularizer=de_conv_regularizer)(y)
        # Crop to 8x8
        y = y[:, 0:8, 0:8]
        y = ReLU()(y)
        decoder = keras.Model([input_dec], y, name="decoder")

        # Full CAE
        cae = Model(
            inputs=[input_enc, cond],
            outputs=decoder([encoder([input_enc, cond])]),
            name="cae"
        )
    else:
        # n_encoded (just an example of an architecture hyperparam you can tune)
        n_encoded = 16

        # -----------------------------------------------------------
        # 3) Bits logic from your code
        # -----------------------------------------------------------
        nIntegerBits = 1
        nDecimalBits = bitsPerOutput - nIntegerBits
        outputSaturationValue = (1 << nIntegerBits) - 1./(1 << nDecimalBits)
        maxBitsPerOutput = 9
        outputMaxIntSize = 1 if (bitsPerOutput <= 0) else (1 << nDecimalBits)

        # Additional fixed settings
        n_kernels = 8
        conv_weightBits = 6
        conv_biasBits   = 6
        dense_weightBits = 6
        dense_biasBits   = 6
        encodedBits = 9
        CNN_kernel_size = 3

        # -----------------------------------------------------------
        # 4) Build Encoder
        # -----------------------------------------------------------
        input_enc = Input(shape=(8, 8, 1), name='Wafer')
        cond      = Input(shape=(8,), name='Cond')

        x = QActivation(quantized_bits(bits=8, integer=1), name='input_quantization')(input_enc)
        x = keras_pad()(x)
        x = QConv2D(
            n_kernels,
            CNN_kernel_size, 
            strides=2, 
            padding='valid',
            kernel_quantizer=quantized_bits(bits=conv_weightBits, integer=0, keep_negative=1, alpha=1),
            bias_quantizer=quantized_bits(bits=conv_biasBits,   integer=0, keep_negative=1, alpha=1),
            name="conv2d"
        )(x)
        x = QActivation(quantized_bits(bits=8, integer=1), name='act')(x)
        x = Flatten()(x)
        x = QDense(
            n_encoded,
            kernel_quantizer=quantized_bits(bits=dense_weightBits, integer=0, keep_negative=1, alpha=1),
            bias_quantizer=quantized_bits(bits=dense_biasBits,     integer=0, keep_negative=1, alpha=1),
            name="dense"
        )(x)

        latent = QActivation(
            qkeras.quantized_bits(bits=encodedBits, integer=nIntegerBits),
            name='latent_quantization'
        )(x)

        # If bits are allocated, rescale + saturate
        if bitsPerOutput > 0 and maxBitsPerOutput > 0:
            latent = keras_floor()(latent * outputMaxIntSize)
            latent = keras_minimum()(latent / outputMaxIntSize, sat_val=outputSaturationValue)

        # Concatenate condition
        latent = concatenate([latent, cond], axis=1)
        encoder = keras.Model([input_enc, cond], latent, name="encoder")

        # -----------------------------------------------------------
        # 5) Build Decoder
        # -----------------------------------------------------------
        input_dec = Input(shape=(24,))
        y = Dense(24)(input_dec)
        y = ReLU()(y)
        y = Dense(64)(y)
        y = ReLU()(y)
        y = Dense(128)(y)
        y = ReLU()(y)
        y = Reshape((4, 4, 8))(y)
        y = Conv2DTranspose(1, (3,3), strides=(2,2), padding='valid')(y)
        # Crop to 8x8
        y = y[:, 0:8, 0:8]
        y = ReLU()(y)
        decoder = keras.Model([input_dec], y, name="decoder")

        # Full CAE
        cae = Model(
            inputs=[input_enc, cond],
            outputs=decoder([encoder([input_enc, cond])]),
            name="cae"
        )
    return cae


##############################################################################
# Learning-Rate Schedulers
##############################################################################


def cos_warm_restarts(epoch, total_epochs, initial_lr):
    """
    Cosine annealing scheduler with warm restarts.
    
    Args:
        epoch        (int): Current epoch.
        total_epochs (int): Total number of epochs.
        initial_lr (float): Initial learning rate.
    
    Returns:
        float: Updated learning rate.
    """
    # We do one 'warm restart' cycle each (total_epochs // 25) epochs
    cos_inner = np.pi * (epoch % (total_epochs // 25))
    cos_inner /= (total_epochs // 25)
    cos_out = np.cos(cos_inner) + 1
    return float(initial_lr / 2 * cos_out)

def cosine_annealing(epoch, total_epochs, initial_lr):
    """
    Cosine annealing scheduler that reduces the learning rate to (initial_lr / 100)
    by the final epoch.
    
    Args:
        epoch        (int): Current epoch.
        total_epochs (int): Total number of epochs.
        initial_lr (float): Initial learning rate.
    
    Returns:
        float: Updated learning rate.
    """
    cos_inner = np.pi * (epoch % total_epochs) / total_epochs
    cos_out = np.cos(cos_inner) + 1
    return float((initial_lr / 2) * cos_out * (1 / 100))

##############################################################################
# Model Parameter Serialization (Example Only)
##############################################################################

def get_pams():
    """
    Example function that serializes some set of parameters.
    It references 'self.pams' and 'telescopeMSE8x8' in the original code,
    which are not defined here. Adjust or remove as needed.
    """
    # This function is incomplete in the snippet and might not work as-is.
    # Provided only for reference.
    raise NotImplementedError("get_pams() depends on external code not provided here.")

##############################################################################
# Model Saving Utilities
##############################################################################

def save_models(autoencoder,model_dir,  name, isQK=False):
    """
    Saves an autoencoder (and its encoder/decoder) in multiple formats:
    JSON, HDF5 weights, frozen graph, and optional QKeras quantized weights.
    
    Args:
        autoencoder (keras.Model): The full autoencoder model to save.
        name              (str): Output file name (without extension).
        isQK            (bool): If True, save QKeras quantized weights.
    """
    # Create the directory if not present

    # JSON Config
    json_string = autoencoder.to_json()

    # Extract sub-models
    encoder = autoencoder.get_layer("encoder")
    decoder = autoencoder.get_layer("decoder")

    # Save models in JSON format
    with open(f"{model_dir}/{name}.json", 'w') as f:
        f.write(json_string)
    with open(f"{model_dir}/encoder_{name}.json", 'w') as f:
        f.write(encoder.to_json())
    with open(f"{model_dir}/decoder_{name}.json", 'w') as f:
        f.write(decoder.to_json())

    # Save weights in HDF5 format
    autoencoder.save_weights(f"{model_dir}/{name}.hdf5")
    encoder.save_weights(f"{model_dir}/encoder_{name}.hdf5")
    decoder.save_weights(f"{model_dir}/decoder_{name}.hdf5")

    # If QKeras, dump quantized weights
    if isQK:
        encoder_qWeight = model_save_quantized_weights(encoder)
        with open(f"{model_dir}/encoder_{name}.pkl", 'wb') as f:
            pickle.dump(encoder_qWeight, f)

        # Re-load them into the encoder model as needed
        encoder = graph.set_quantized_weights(encoder, f"{model_dir}/encoder_{name}.pkl")

    # Save frozen graphs (encoder, decoder)
    graph.write_frozen_graph_enc(encoder,  f"encoder_{name}.pb",       logdir=model_dir)
    graph.write_frozen_graph_enc(encoder,  f"encoder_{name}.pb.ascii", logdir=model_dir, asText=True)
    graph.write_frozen_graph_dec(decoder,  f"decoder_{name}.pb",       logdir=model_dir)
    graph.write_frozen_graph_dec(decoder,  f"decoder_{name}.pb.ascii", logdir=model_dir, asText=True)

    # Plot weights for debugging/inspection
    graph.plot_weights(autoencoder, outdir=model_dir)
    graph.plot_weights(encoder,     outdir=model_dir)
    graph.plot_weights(decoder,     outdir=model_dir)

def save_CMSSW_models(encoder, decoder, model_dir,  name, isQK=False):
    
    if isQK:
        encoder_qWeight = model_save_quantized_weights(encoder)
        with open(f'{model_dir}/encoder_{name}.pkl','wb') as f:
            pickle.dump(encoder_qWeight,f)
    graph.write_frozen_dummy_enc(encoder,'encoder_'+name+'.pb',logdir = model_dir)
    graph.write_frozen_graph_dec(decoder,'decoder_'+name+'.pb',logdir = model_dir)

##############################################################################
# Example Weighted MSE Loss
##############################################################################

# If you want to reorder the 8x8 in the same manner as your original code:
remap_8x8 = [
    4, 12, 20, 28,  5, 13, 21, 29,
    6, 14, 22, 30,  7, 15, 23, 31,
    24, 25, 26, 27, 16, 17, 18, 19,
    8,  9, 10, 11,  0,  1,  2,  3,
    59, 51, 43, 35, 58, 50, 42, 34,
    57, 49, 41, 33, 56, 48, 40, 32
]

def mean_mse_loss(y_true, y_pred):
    """
    Example MSE loss function that re-maps the 8x8 patch and
    weights by the maximum value in y_true.
    
    Args:
        y_true (tf.Tensor): True wafer images.
        y_pred (tf.Tensor): Predicted wafer images.
    
    Returns:
        tf.Tensor: Scalar loss.
    """
    # Weighted by the maximum values in the ground truth
    max_values = tf.reduce_max(y_true, axis=1)

    # Reshape y_true and y_pred to (batch, 64), then gather specific indices
    y_true = tf.gather(K.reshape(y_true, (-1, 64)), remap_8x8, axis=-1)
    y_pred = tf.gather(K.reshape(y_pred, (-1, 64)), remap_8x8, axis=-1)

    squared_diff = tf.square(y_pred - y_true)
    mse_per_row = tf.reduce_mean(squared_diff, axis=1)
    weighted_mse_per_row = mse_per_row * max_values

    return tf.reduce_mean(weighted_mse_per_row)

##############################################################################
# Resampling Utilities
##############################################################################

def resample_indices(indices, energy, bin_edges, target_count, bin_index):
    """
    Resample indices based on bin definitions and target counts.
    
    Args:
        indices      (np.ndarray): Indices array.
        energy       (np.ndarray): Energy array for each index.
        bin_edges    (list/array): Edges defining bins.
        target_count      (int): Target number of samples in each bin.
        bin_index        (int): Which bin we're dealing with.
    
    Returns:
        np.ndarray: Resampled indices.
    """
    bin_indices = indices[
        (energy > bin_edges[bin_index]) & 
        (energy <= bin_edges[bin_index+1])
    ]
    if len(bin_indices) > target_count:
        return np.random.choice(bin_indices, size=target_count, replace=False)
    else:
        return np.random.choice(bin_indices, size=target_count, replace=True)

def custom_resample(wafers, c, simE, args):
    """
    Example function that oversamples and/or undersamples based on a 'biased' fraction
    in args. Adjust as needed for your environment.
    
    Args:
        wafers (np.ndarray): Wafers array of shape [N, 8, 8, 1].
        c      (np.ndarray): Some condition array of shape [N, ...].
        simE   (np.ndarray): Array with energies [N, ...].
        args        (obj): Object containing hyperparameters (e.g., args.biased).
    
    Returns:
        (wafers_p, c_p): Resampled wafer data.
    """
    label = (simE[:, 0] != 0).astype(int)
    n = len(label)
    print("Original label distribution:", Counter(label))
    indices = np.expand_dims(np.arange(n), axis=-1)

    # 10x upsample signal if 'biased' < 0.9
    if args.biased < 0.9:
        over = RandomOverSampler(sampling_strategy=0.1)
        indices_p, label_p = over.fit_resample(indices, label)
    else:
        indices_p, label_p = indices, label

    # Downsample until ratio or 1:2 for (pileup : signal)
    signal_percent = 1 - args.biased
    ratio = args.biased / signal_percent
    if ratio > 1:
        # If ratio>1, invert the ratio for under-sampling
        ratio = 1 / ratio
        under = RandomUnderSampler(sampling_strategy=ratio)
        indices_p, label_p = under.fit_resample(indices_p, label_p)
    else:
        under = RandomUnderSampler(sampling_strategy=ratio)
        indices_p, label_p = under.fit_resample(indices_p, label_p)

    print("Resampled label distribution:", Counter(label_p))

    wafers_p = wafers[indices_p[:, 0]]
    c_p      = c[indices_p[:, 0]]
    return wafers_p, c_p

##############################################################################
# Masking Utility for 'Old Geometry'
##############################################################################

def get_old_mask(eLinks, df):
    """
    Generate a boolean mask for a given DataFrame 'df' based on 'eLinks'
    and the 'layer' column. Adjust as needed for your geometry logic.
    
    Args:
        eLinks (int): Number of eLinks (used to pick certain layers).
        df   (pandas.DataFrame): DataFrame with a 'layer' column.
    
    Returns:
        pandas.Series of booleans: True for rows that pass the mask, False otherwise.
    """
    mask = pd.Series([False] * len(df), index=df.index)

    if eLinks == 5:
        mask |= ((df['layer'] <= 11) & (df['layer'] >= 5))
    elif eLinks == 4:
        mask |= ((df['layer'] == 7) | (df['layer'] == 11))
    elif eLinks == 3:
        mask |= (df['layer'] == 13)
    elif eLinks == 2:
        mask |= ((df['layer'] < 7) | (df['layer'] > 13))
    elif eLinks == -1:
        mask |= (df['layer'] > 0)

    return mask

##############################################################################
# Data Loading
##############################################################################

def load_pre_processed_data(nfiles, batchsize, bits, args):
    """
    Example data loader. Assumes TFRecord-like datasets are stored under
    args.data_path with subfolders named 'data_{bits}_eLinks'.
    
    Args:
        nfiles    (int): Number of files to load.
        batchsize (int): Batch size for training.
        bits      (int): eLinks or bits parameter for subfolder selection.
        args    (obj): Object containing dataset paths & sizes (train/val/test).
    
    Returns:
        (train_loader, test_loader, val_loader): tf.data.Dataset objects.
    """
    data_folder = os.path.join(args.data_path, f"data_{bits}_eLinks")
    files = os.listdir(data_folder)
    
    # Separate training & testing files
    train_files = [f for f in files if "train" in f][:nfiles]
    test_files  = [f for f in files if "test"  in f][:nfiles]
    

    # Combine all training files
    train_datasets = []
    for file in train_files:
        ds = tf.data.experimental.load(os.path.join(data_folder, file))
        train_datasets.append(ds)
    train_dataset = train_datasets[0]
    for ds in train_datasets[1:]:
        train_dataset = train_dataset.concatenate(ds)

    # Combine all testing files
    test_datasets = []
    for file in test_files:
        ds = tf.data.experimental.load(os.path.join(data_folder, file))
        print('len ds :',len(list(ds.as_numpy_iterator())))
        test_datasets.append(ds)
    test_dataset = test_datasets[0]
    
    for ds in test_datasets[1:]:
        test_dataset = test_dataset.concatenate(ds)
    print('len test_dataset :',len(list(test_dataset.as_numpy_iterator())))
    # Optionally subset
    train_size = args.train_dataset_size
    val_size = args.val_dataset_size
    test_size = args.test_dataset_size
    # print('test_size:',test_size)
    # Ensure combined train and val size is not larger than the original train_dataset
    total_train_val_size = train_size + val_size
    original_train_size = len(train_dataset)

    if total_train_val_size > original_train_size:
        train_size = int(original_train_size * (train_size / total_train_val_size))
        val_size = original_train_size - train_size

    train_dataset = train_dataset.take(train_size)
    val_dataset = train_dataset.skip(train_size).take(val_size)

    # Ensure test size is not larger than the original test_dataset
    original_test_size = len(list(test_dataset.as_numpy_iterator()))
    if test_size > original_test_size:
        test_size = original_test_size

    test_dataset = test_dataset.take(test_size)
    print('len test_dataset :',len(list(test_dataset.as_numpy_iterator())))
    # Prepare the data loaders
    train_loader = train_dataset.batch(batchsize)
    test_loader  = test_dataset.batch(batchsize)
    val_loader   = val_dataset.batch(batchsize)

    return train_loader, test_loader, val_loader


def load_pre_processed_data_for_hyperband(nfiles, bits, args, dataset_limit = 250_000):
    """
    Example data loader. Assumes TFRecord-like datasets are stored under
    args.data_path with subfolders named 'data_{bits}_eLinks'.
    
    Args:
        nfiles    (int): Number of files to load.
        batchsize (int): Batch size for training.
        bits      (int): eLinks or bits parameter for subfolder selection.
        args    (obj): Object containing dataset paths & sizes (train/val/test).
    
    Returns:
        (train_loader, test_loader, val_loader): tf.data.Dataset objects.
    """
    data_folder = os.path.join(args.data_path, f"data_{bits}_eLinks")
    files = os.listdir(data_folder)
    
    # Separate training & testing files
    train_files = [f for f in files if "train" in f][:nfiles]
    test_files  = [f for f in files if "test"  in f][:nfiles]

    # Combine all training files
    train_datasets = []
    for file in train_files:
        ds = tf.data.experimental.load(os.path.join(data_folder, file))
        train_datasets.append(ds)
    train_dataset = train_datasets[0]
    for ds in train_datasets[1:]:
        train_dataset = train_dataset.concatenate(ds)

    # Combine all testing files
    test_datasets = []
    for file in test_files:
        ds = tf.data.experimental.load(os.path.join(data_folder, file))
        test_datasets.append(ds)
    test_dataset = test_datasets[0]
    for ds in test_datasets[1:]:
        test_dataset = test_dataset.concatenate(ds)

    # Optionally subset
    total_size = len(list(train_dataset))  # Ensure we get the correct dataset size

    # Limit dataset size to prevent excessive memory usage
    total_size = min(total_size, dataset_limit)

    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = int(0.1 * total_size)

    # Ensure train + val sizes do not exceed available samples
    if train_size + val_size > total_size:
        train_size = int(total_size * (train_size / (train_size + val_size)))
        val_size = total_size - train_size

    # Split datasets properly
    all_train_dataset = train_dataset
    all_train_dataset = all_train_dataset.shuffle(buffer_size = total_size, seed=42)


    # Assuming total_size is defined
    buffer_size = total_size  # Adjust based on your memory capacity
    num_parallel_calls = tf.data.experimental.AUTOTUNE  # Automatically chooses the best number of parallel calls

    all_train_dataset = train_dataset
    all_train_dataset = all_train_dataset.shuffle(buffer_size=buffer_size, seed=42)
    all_train_dataset = all_train_dataset.prefetch(buffer_size)

    del train_dataset

    train_dataset = all_train_dataset.take(train_size)
    val_dataset = all_train_dataset.skip(train_size).take(val_size)
    

    # Debugging output
    print('TESTING DATA LOADING')
    print('Train size:', len(list(train_dataset.as_numpy_iterator())))  # Convert to list to count elements
    print('Val size:', len(list(val_dataset.as_numpy_iterator())))

    # Ensure test size is not larger than the original test_dataset
    original_test_size = tf.data.Dataset.cardinality(test_dataset)
    if test_size > original_test_size:
        test_size = original_test_size

    test_dataset = test_dataset.take(test_size)

    

    return train_dataset, test_dataset, val_dataset
##############################################################################
# Directory Management
##############################################################################

def makedir(outdir, continue_training=False):
    """
    Creates the output directory. If it already exists (and continue_training is False),
    appends a timestamp to make it unique.
    
    Args:
        outdir (str): Desired output directory.
        continue_training (bool): If True, allows existing directory usage.
    
    Returns:
        str: Possibly updated directory name.
    """
    if os.path.isdir(outdir) and not continue_training:
        now = datetime.now()
        # Example date/time formatting:
        outdir += now.strftime("%Y_%m_%d_%H_%M")
    os.system("mkdir -p " + outdir)
    return outdir

##############################################################################
# Command-Line Argument Parser
##############################################################################

class Opt(dict):
    """
    Dictionary-based class for custom store actions with
    typed arguments (INT, FLOAT, STR, etc.).
    """
    def __init__(self, *args, **kwargs):
        super(Opt, self).__init__()
        for a in args:
            if isinstance(a, dict):
                self.update(a)
        self.update(kwargs)

    def __add__(self, other):
        return Opt(self, other)

    def __iadd__(self, other):
        self.update(other)
        return self


class ArgumentParser(_AP):
    STORE_TRUE = Opt({'action':'store_true'})
    STORE_FALSE = Opt({'action':'store_false'})
    MANY = Opt({'nargs':'+'})
    INT = Opt({'type': int})
    FLOAT = Opt({'type': float})
    STR = Opt({'type': str})

    class Namespace(object):
        def __init__(self):
            pass

        def save_to(self, path):
            yaml.dump({k:getattr(self, k) for k in vars(self)},
                      open(path, 'w'),
                      default_flow_style=True)

        def __str__(self):
            return str({k:getattr(self, k) for k in vars(self)})

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        super().add_argument('-c', '--config', nargs='+', default=[])

    def add_arg(self, *args, **kwargs):
        if 'default' in kwargs:
            logger.error(f'default is not allowed in ArgumentParser')
            raise RuntimeError()
        return super().add_argument(*args, **kwargs)

    def add_args(self, *args):
        for a in args:
            if type(a) == tuple:
                self.add_arg(a[0], **a[1])
            else:
                self.add_arg(a)

    def parse_args(self, *args, **kwargs):
        cmd_line_args = super().parse_args(*args, **kwargs)
        args = ArgumentParser.Namespace()
        for k in vars(cmd_line_args):
            v = getattr(cmd_line_args, k)
            setattr(args, k, v)
        for conf in cmd_line_args.config:
            payload = yaml.safe_load(open(conf, 'r'))
            for k,v in payload.items():
                setattr(args, k, v)
                logger.debug(f'Config {conf} : {k} -> {v}')
        self.args = args
        return args

##############################################################################
# Snapshot Management
##############################################################################

class Snapshot(object):
    """
    Example class for handling experiment snapshots (logging, saving config, etc.).
    Adjust or remove as needed based on your environment.
    """
    def __init__(self, base_path, args):
        if hasattr(args, 'checkpoint_path'):
            self.path = args.checkpoint_path
        else:
            self.path = os.path.join(base_path, time.strftime("%Y_%m_%d_%H_%M_%S"))
        print(f"Snapshot placed at {self.path}")

        if not os.path.exists(self.path):
            os.makedirs(self.path)

        uuid_str = str(uuid4())
        self.args = args
        args.save_to(self.get_path(uuid_str + '.args.yaml'))

        # If you have a logger, set it up here. 
        # This example simply prints to stdout.
        # Example:
        # logger.add(self.get_path(uuid_str + '.snapshot.log'), level='DEBUG')

    def get_path(self, filename):
        return os.path.join(self.path, filename)

##############################################################################
# Miscellaneous Helpers
##############################################################################

def t2n(t):
    """
    If tensor, return .numpy(); else, return as-is (or None).
    """
    if t is None:
        return None
    if isinstance(t, np.ndarray):
        return t
    return t.numpy()

def rescore(yhat, q, y, rescale=True):
    """
    Example post-processing function. 
    If q != 0, override predictions with ground truth (y).
    
    Args:
        yhat   (np.ndarray): predicted values
        q      (np.ndarray): mask or condition array
        y      (np.ndarray): ground truth
        rescale    (bool): whether to scale yhat in range [0, 1] for q==0
        
    Returns:
        np.ndarray: rescaled or replaced array
    """
    if rescale:
        q_mask = (q == 0)
        lo = yhat[q_mask].min()
        hi = yhat[q_mask].max()
        yhat[q_mask] = (yhat[q_mask] - lo) / (hi - lo + 1e-9)

    q_mask = (q != 0)
    yhat[q_mask] = y[q_mask]
    return yhat

##############################################################################
# Encoding / Decoding Utilities
##############################################################################

def encode(value, dropBits=1, expBits=4, mantBits=3, roundBits=False, asInt=False):
    """
    Example integer-based encoding with a floating exponent/mantissa approach.
    Often used for approximate compression of integer data.
    
    Args:
        value       (int): Input integer value to encode.
        dropBits    (int): How many bits to drop (shifts right by dropBits).
        expBits     (int): Number of bits used for exponent.
        mantBits    (int): Number of bits used for mantissa.
        roundBits  (bool): If True, rounds value when dropping bits.
        asInt      (bool): If True, returns an integer; else returns a string.
    
    Returns:
        (int or str): The encoded representation.
    """
    bin_code = bin(value)[2:]
    
    # If the total length is small enough, simple case
    if len(bin_code) <= (mantBits + dropBits):
        if roundBits and dropBits > 0:
            value += 2 ** (dropBits - 1)
        value = value >> dropBits
        bin_code = bin(value)[2:]
        mantissa = format(value, '#0%ib' % (mantBits + 2))[2:]
        exponent = '0' * expBits
    elif len(bin_code) == (mantBits + dropBits + 1):
        if roundBits and dropBits > 0:
            value += 2 ** (dropBits - 1)
        value = value >> dropBits
        bin_code = bin(value)[2:]
        exponent = '0001'
        mantissa = bin_code[1:1 + mantBits]
    else:
        if roundBits:
            v_temp = int(bin_code, 2) + int(2 ** (len(bin_code) - 2 - mantBits))
            bin_code = bin(v_temp)[2:]
        first_zero = len(bin_code) - mantBits - dropBits
        if first_zero < 1:
            raise ValueError("Invalid encoding: insufficient length.")
        if first_zero < 2 ** expBits:
            exponent = format(first_zero, '#0%ib' % (expBits + 2))[2:]
            mantissa = bin_code[1:1 + mantBits]
        else:
            exponent = '1' * expBits
            mantissa = '1' * mantBits

    if asInt:
        return int(exponent + mantissa, 2)
    else:
        return exponent + mantissa

def decode(val, droppedBits=1, expBits=4, mantBits=3, edge=False, quarter=False):
    """
    Example decoder for values generated by the above 'encode' method.
    
    Args:
        val         (int): Encoded value (e.g. exponent+mantissa bits).
        droppedBits (int): How many bits were dropped (shift in reconstruction).
        expBits     (int): Number of bits used for exponent.
        mantBits    (int): Number of bits used for mantissa.
        edge       (bool): If True, modifies how rounding is done.
        quarter    (bool): Another shift-based rounding option.
    
    Returns:
        (int): Decoded integer approximation of the original input.
    """
    exp = val >> mantBits
    mant = val & (2**mantBits - 1)

    # Build up the reconstructed data
    if exp > 0:
        data = (mant << (exp - 1)) + (1 << (exp + mantBits - 1))
    else:
        data = mant

    data = data << droppedBits
    shift = max(exp - 1, 0)

    if quarter:
        if (droppedBits + shift) > 1:
            data += 1 << (shift + droppedBits - 2)
    elif not edge:
        if (droppedBits + shift) > 0:
            data += 1 << (shift + droppedBits - 1)

    return data
