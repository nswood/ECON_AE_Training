import os
import pickle
import numpy as np
import pandas as pd

import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
    # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
          tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
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
from telescope import telescopeMSE8x8
from utils import (ArgumentParser, load_pre_processed_data, mean_mse_loss, 
                   cos_warm_restarts, cosine_annealing, 
                   save_models)

################################################################
# Custom Keras Layers
################################################################

class keras_pad(Layer):
    """
    Custom zero-padding layer. Pads the incoming tensor with zeros
    on the bottom and right edges.
    """
    def call(self, x):
        # [batch, height, width, channels] -> pad height & width by 1 each
        padding = tf.constant([[0, 0], [0, 1], [0, 1], [0, 0]])
        return tf.pad(x, padding, mode='CONSTANT', constant_values=0)

class keras_minimum(Layer):
    """
    Custom layer to apply element-wise minimum operation between
    the input and a saturation value 'sat_val'.
    """
    def call(self, x, sat_val=1):
        return tf.minimum(x, sat_val)

class keras_floor(Layer):
    """
    Custom floor operation for dense or sparse tensors.
    """
    def call(self, x):
        if isinstance(x, tf.SparseTensor):
            x = tf.sparse.to_dense(x)
        return tf.math.floor(x)

################################################################
# Parse Command-Line Arguments
################################################################


p = ArgumentParser()

p = ArgumentParser()

# Paths
p.add_argument('--opath', type=str, required=True)
p.add_argument('--mpath', type=str, required=False)

# Model parameters
p.add_argument('--mname', type=str, required=True)
p.add_argument('--model_per_eLink', action='store_true')
p.add_argument('--model_per_bit_config', action='store_true')
p.add_argument('--alloc_geom', type=str, required=True)
p.add_argument('--specific_m', type=int, required=False)

# Training parameters
p.add_argument('--continue_training', action='store_true')
p.add_argument('--loss', type=str, default='tele')
p.add_argument('--lr', type=float, required=True)
p.add_argument('--nepochs', type=int, required=True)
p.add_argument('--batchsize', type=int, required=True)
p.add_argument('--optim', type=str, choices=['adam', 'lion'], default='lion')
p.add_argument('--lr_scheduler', type=str, default='cos')

# Dataset parameters
p.add_argument('--data_path', type=str, required=True)
p.add_argument('--num_files', type=int, required=True)
p.add_argument('--train_dataset_size', type=int, default=500000)
p.add_argument('--val_dataset_size', type=int, default=100000)
p.add_argument('--test_dataset_size', type=int, default=100000)

args = p.parse_args()

################################################################
# Create and Verify Output Directory
################################################################

output_dir = os.path.join(args.opath, 'training_models')
if not os.path.exists(output_dir):
    os.system("mkdir -p " + args.opath)
    os.system("mkdir -p " + output_dir)

################################################################
# Determine Model(s) to Train
################################################################

if args.specific_m is not None:
    all_models = [args.specific_m]
elif args.model_per_eLink:
    if args.alloc_geom == 'old':
        all_models = [2, 3, 4, 5]
    elif args.alloc_geom == 'new':
        all_models = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
elif args.model_per_bit_config:
    if args.alloc_geom == 'old':
        all_models = [3, 5, 7, 9]
    elif args.alloc_geom == 'new':
        all_models = [1, 3, 5, 7, 9]

bitsPerOutputLink = [
    0,  1,  3,  5,  7,  9, 
    9,  9,  9,  9,  9,  9, 
    9,  9,  9
]    

################################################################
# Main Training Loop over the eLink or Bit Configurations
################################################################

for m in all_models:
    # Determine bits for output or number of eLinks
    if args.model_per_eLink:
        eLinks = m
        bitsPerOutput = bitsPerOutputLink[eLinks]
        print(f"Training Model with {eLinks} eLinks")
        model_dir = os.path.join(output_dir, f"model_{eLinks}_eLinks")
    elif args.model_per_bit_config:
        bitsPerOutput = m
        print(f"Training Model with {bitsPerOutput} output bits")
        model_dir = os.path.join(output_dir, f"model_{bitsPerOutput}_bits")
    
    if not os.path.exists(model_dir):
        os.system("mkdir -p " + model_dir)

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

    ################################################################
    # Select Loss Function
    ################################################################

    if args.loss == 'mse':
        loss_fn = mean_mse_loss
    elif args.loss == 'tele':
        print('Using telescope MSE (8x8) loss')
        loss_fn = telescopeMSE8x8
    elif args.loss == 'emd':
        loss_fn = get_emd_loss(args.emd_pth)
    else:
        raise ValueError("Unknown loss function specified.")

    ################################################################
    # Optimizer Setup
    ################################################################

    if args.optim == 'adam':
        print('Using ADAM Optimizer')
        opt = tf.keras.optimizers.Adam(learning_rate=args.lr, weight_decay=0.000025)
    elif args.optim == 'lion':
        print('Using Lion Optimizer')
        opt = tf.keras.optimizers.Lion(learning_rate=args.lr, weight_decay=0.00025)
    else:
        raise ValueError("Unknown optimizer specified.")

    cae.compile(optimizer=opt, loss=loss_fn)
    cae.summary()

    ################################################################
    # Learning-Rate Scheduler
    ################################################################

    initial_lr   = args.lr
    total_epochs = args.nepochs

    # We demonstrate using only the 'cos' scheduler below
    # (you can still switch to cos_warm_restarts if needed)
    if args.lr_scheduler == 'cos_warm_restarts':
        lr_schedule = lambda epoch: cos_warm_restarts(
            epoch, total_epochs=total_epochs, initial_lr=initial_lr
        )
    elif args.lr_scheduler == 'cos':
        lr_schedule = lambda epoch: cosine_annealing(
            epoch, total_epochs=total_epochs, initial_lr=initial_lr
        )
    else:
        raise ValueError("Unknown LR scheduler specified.")

    lr_scheduler_cb = tf.keras.callbacks.LearningRateScheduler(lr_schedule)
    print(f"Training with {args.lr_scheduler} scheduler")

    ################################################################
    # Optional: Continue Training
    ################################################################

    best_val_loss = 1e9
    if args.continue_training:
        # Load existing model weights
        cae.load_weights(os.path.join(model_dir, 'best-epoch.tf'))
        print("Continuing training from saved best model...")

    ################################################################
    # Load Data
    ################################################################

    print('Loading Data...')
    train_loader, test_loader, val_loader = load_pre_processed_data(args.num_files, batch, m, args)
    print('Data Loaded!')

    # Dump dataset sizes and arguments to a text file
    with open(os.path.join(model_dir, 'training_info.txt'), 'w') as f:
        f.write(f"Training dataset size: {len(train_loader) * args.batchsize}\n")
        f.write(f"Validation dataset size: {len(val_loader)* args.batchsize }\n")
        f.write(f"Test dataset size: {len(test_loader)* args.batchsize}\n")
        f.write("Arguments:\n")
        for arg in vars(args):
            f.write(f"{arg}: {getattr(args, arg)}\n")

    # To store loss progression
    if args.continue_training and os.path.exists(os.path.join(model_dir, 'df.csv')):
        df_existing = pd.read_csv(os.path.join(model_dir, 'df.csv'))
        loss_dict = {
            'train_loss': df_existing['train_loss'].tolist(),
            'val_loss':   df_existing['val_loss'].tolist()
        }
        start_epoch = len(loss_dict['train_loss']) + 1
    else:
        start_epoch = 1
        loss_dict = {'train_loss': [], 'val_loss': []}

    ################################################################
    # Training Loop
    ################################################################

    for epoch in range(start_epoch, total_epochs + 1):

        # Adjust learning rate via the chosen schedule
        new_lr = lr_schedule(epoch)
        tf.keras.backend.set_value(opt.learning_rate, new_lr)

        # ----------- Training -----------
        total_loss_train = 0
        for wafers, cond_data in train_loader:
            loss_batch = cae.train_on_batch([wafers, cond_data], wafers)
            total_loss_train += loss_batch

        # ----------- Validation -----------
        total_loss_val = 0
        for wafers, cond_data in test_loader:
            loss_batch_val = cae.test_on_batch([wafers, cond_data], wafers)
            total_loss_val += loss_batch_val

        # Average across all batches
        total_loss_train /= len(train_loader)
        total_loss_val   /= len(test_loader)

        print(f"Epoch {epoch:03d}, "
              f"Loss: {total_loss_train:.8f}, "
              f"ValLoss: {total_loss_val:.8f}")

        # Log the losses
        loss_dict['train_loss'].append(total_loss_train)
        loss_dict['val_loss'].append(total_loss_val)
        df_log = pd.DataFrame.from_dict(loss_dict)

        # Save training curves
        plt.figure(figsize=(10, 6))
        plt.plot(df_log['train_loss'], label='Training Loss')
        plt.plot(df_log['val_loss'],   label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.yscale('log')
        plt.legend()
        plt.grid(True)
        plot_path = f"{model_dir}/training_loss_plot.png"
        plt.savefig(plot_path)
        plt.close()

        # Save CSV log
        df_log.to_csv(f"{model_dir}/df.csv", index=False)

        # Save best model
        if total_loss_val < best_val_loss:
            print("New Best Model Found!")
            best_val_loss = total_loss_val
            cae.save_weights(os.path.join(model_dir, 'best-epoch.tf'))
            encoder.save_weights(os.path.join(model_dir, 'best-encoder-epoch.tf'))
            decoder.save_weights(os.path.join(model_dir, 'best-decoder-epoch.tf'))

    ################################################################
    # Post-Training: Save Entire Model
    ################################################################

    # This function presumably exports your QKeras model for external use
    save_models(cae, model_dir, args.mname, isQK=True)

################################################################
# Call the dev_preCMSSW Script with Updated Args
################################################################

import subprocess

if args.model_per_eLink:
    script_args = [
        'python', 'preprocess_CMSSW.py',
        '--mname', 'vanilla_AE',
        '--mpath', args.opath,
        '--model_per_eLink',
        '--alloc_geom', args.alloc_geom,
    ]
elif args.model_per_bit_config:
    script_args = [
        'python', 'preprocess_CMSSW.py',
        '--mname', 'vanilla_AE',
        '--mpath', args.opath,
        '--model_per_bit_config',
        '--alloc_geom', args.alloc_geom,
    ]
else:
    script_args = []

# Run the other script with the arguments (only if relevant flags are set)
if script_args:
    subprocess.run(script_args)