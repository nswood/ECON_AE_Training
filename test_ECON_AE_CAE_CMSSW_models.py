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
# Helper Function to Load Frozen Graphs from .pb Files
################################################################
def load_frozen_graph(pb_path):
    with tf.io.gfile.GFile(pb_path, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")  # name="" to keep original tensor names
    return graph

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

# Setup fixed parameters (as in your original code)

nIntegerBits = 1
nDecimalBits = bitsPerOutput - nIntegerBits
outputSaturationValue = (1 << nIntegerBits) - 1. / (1 << nDecimalBits) if bitsPerOutput > 0 else 1
maxBitsPerOutput = 9

outputMaxIntSize = (1 << nDecimalBits) if bitsPerOutput > 0 else 1
outputMaxIntSizeGlobal = (1 << (maxBitsPerOutput - nIntegerBits)) if maxBitsPerOutput > 0 else 1

# (Optional) The following block builds the models in Keras.
# You can keep this for reference but will use the frozen graphs below.
##########################################################################
# Build Encoder (for reference)
##########################################################################
input_enc = keras.Input(batch_shape=(batch, 8, 8, 1), name='Wafer')
x = QActivation(quantized_bits(bits=8, integer=1), name='input_quantization')(input_enc)
padding = tf.constant([[0, 0], [0, 1], [0, 1], [0, 0]])
x = tf.pad(x, padding, mode='CONSTANT', constant_values=0)
x = QConv2D(
    8, 3, strides=2, padding='valid',
    kernel_quantizer=quantized_bits(bits=6, integer=0, keep_negative=1, alpha=1),
    bias_quantizer=quantized_bits(bits=6, integer=0, keep_negative=1, alpha=1),
    name="conv2d"
)(x)
x = QActivation(quantized_bits(bits=8, integer=1), name='act')(x)
x = Flatten()(x)
x = QDense(
    16,
    kernel_quantizer=quantized_bits(bits=6, integer=0, keep_negative=1, alpha=1),
    bias_quantizer=quantized_bits(bits=6, integer=0, keep_negative=1, alpha=1),
    name="dense"
)(x)
x = QActivation(quantized_bits(bits=9, integer=1), name='latent_quantization')(x)
if bitsPerOutput > 0 and maxBitsPerOutput > 0:
    x_floor = tf.math.floor(x * outputMaxIntSize)
    x = tf.minimum(x_floor / outputMaxIntSize, outputSaturationValue)
encoder_keras = keras.Model([input_enc], x, name="encoder")

##########################################################################
# Build Decoder (for reference)
##########################################################################
input_dec = keras.Input(batch_shape=(batch, 24), name='input_dec')
y = Dense(24)(input_dec)
y = ReLU()(y)
y = Dense(64)(y)
y = ReLU()(y)
y = Dense(128)(y)
y = ReLU()(y)
y = Reshape((4, 4, 8))(y)
y = Conv2DTranspose(1, (3, 3), strides=(2, 2), padding='valid')(y)
y = y[:, 0:8, 0:8]  # Slice to (8x8)
y = ReLU()(y)
decoder_keras = keras.Model([input_dec], y, name="decoder")

# ---------------------------------------------------------------------
# Instead of loading using tf.saved_model.load, load the frozen graphs:
encoder_pb_path = os.path.join(model_dir, 'encoder_model.pb')
decoder_pb_path = os.path.join(model_dir, 'decoder_model.pb')

encoder_graph = load_frozen_graph(encoder_pb_path)
decoder_graph = load_frozen_graph(decoder_pb_path)
# def print_graph_tensors(graph):
#     for op in graph.get_operations():
#         print("Operation:", op.name)
#         for tensor in op.outputs:
#             print("    Tensor:", tensor.name)

# Example usage for the encoder graph:
# encoder_graph = load_frozen_graph(encoder_pb_path)
# print("Encoder Graph Tensors:")
# print_graph_tensors(encoder_graph)

# Similarly, for the decoder graph:
# decoder_graph = load_frozen_graph(decoder_pb_path)
# print("Decoder Graph Tensors:")
# print_graph_tensors(decoder_graph)
# Set up sessions for each frozen graph.
# Replace the tensor names below with the actual names in your frozen graphs.
try:
    # For the encoder graph:
    encoder_input_tensor = encoder_graph.get_tensor_by_name("x:0")
    encoder_output_tensor = encoder_graph.get_tensor_by_name("encoder/tf.math.minimum/Minimum:0")
except Exception as e:
    print("Error retrieving encoder tensor names:", e)
    

try:
    # For the decoder graph:
    decoder_input_tensor = decoder_graph.get_tensor_by_name("x:0")
    # For example, assume the decoder's final output tensor is the output of the Reshape layer.
    decoder_output_tensor = decoder_graph.get_tensor_by_name("decoder/re_lu_3/Relu:0")
except Exception as e:
    print("Error retrieving decoder tensor names:", e)
    
    
print('Loading Data...')
train_loader, test_loader, val_loader = load_pre_processed_data(args.num_files, batch, m, args)
print('Data Loaded!')

################################################################
# Inference and Validation using the Frozen Graphs
################################################################
data_batches = list(test_loader.as_numpy_iterator())
if len(data_batches) > 100000:
    data_batches = data_batches[:100000]
print('len(data_batches):', len(data_batches))
total_loss_val = 0.0
num_batches = 0
all_output = []
all_inputs = [] 
with tf.compat.v1.Session(graph=encoder_graph) as enc_sess, \
    tf.compat.v1.Session(graph=decoder_graph) as dec_sess:
    for wafers, cond_data in data_batches:
        # Run encoder inference.
        latent = enc_sess.run(encoder_output_tensor, feed_dict={encoder_input_tensor: wafers})
        # Concatenate latent representation with cond_data.
        latent_concat = np.concatenate([latent, cond_data], axis=1)
        # Run decoder inference.
        output = dec_sess.run(decoder_output_tensor, feed_dict={decoder_input_tensor: latent_concat})
        all_output.append(output)
        all_inputs.append(wafers)
        num_batches += 1
        
        # Compute loss.
all_inputs = np.concatenate(all_inputs, axis=0)
all_output = np.concatenate(all_output, axis=0)
total_loss_val = telescopeMSE8x8(all_inputs, all_output)
total_loss_val = total_loss_val.numpy().sum()
print('Total Loss:', total_loss_val)
print('Type of total_loss_val:', type(total_loss_val))
if num_batches > 0:
    total_loss_val /= num_batches


print(f"Test Loss: {float(total_loss_val):.8f}")

