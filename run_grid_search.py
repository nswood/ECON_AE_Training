import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools

import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
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

# Custom utilities
from utils.telescope import telescopeMSE8x8
from utils.utils import *
################################################################
# Custom Keras Layers
################################################################
class keras_pad(Layer):
    """
    Custom zero-padding layer. Pads the incoming tensor with zeros
    on the bottom and right edges.
    """
    def call(self, x):
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
# Training function for a single set of hyperparameters
################################################################
def train_single_configuration(args, model_dir):
    """
    Trains a single configuration (model architecture, training loop)
    and saves the best epoch by val_loss into `model_dir`.
    """

    # -----------------------------------------------------------
    # Create and verify output directory
    # -----------------------------------------------------------
    # We'll create a "training_models" subdir within model_dir to store
    # the logs, best weights, etc.
    output_dir = os.path.join(model_dir, 'training_models')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # -----------------------------------------------------------
    # Which model(s) to train
    # -----------------------------------------------------------
    all_models = [args.specific_m]

    bitsPerOutputLink = [
        0, 1, 3, 5, 7, 9, 
        9, 9, 9, 9, 9, 9, 
        9, 9, 9
    ]    

    for m in all_models:
        # Decide if we train per eLink or per bit config
        if args.model_per_eLink:
            eLinks = m
            bitsPerOutput = bitsPerOutputLink[eLinks]
            print(f"Training Model with {eLinks} eLinks")
            sub_dir = os.path.join(output_dir, f"model_{eLinks}_eLinks")
        elif args.model_per_bit_config:
            bitsPerOutput = m
            print(f"Training Model with {bitsPerOutput} output bits")
            sub_dir = os.path.join(output_dir, f"model_{bitsPerOutput}_bits")
        else:
            # fallback if no mode is specified
            bitsPerOutput = m
            sub_dir = os.path.join(output_dir, f"model_{bitsPerOutput}_bits_default")

        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir, exist_ok=True)

        # -------------------------------------------------------
        # Model Hyperparams
        # -------------------------------------------------------
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

        # -------------------------------------------------------
        # Encoder Definition
        # -------------------------------------------------------
        input_enc = Input(batch_shape=(batch, 8, 8, 1), name='Wafer')
        cond      = Input(batch_shape=(batch, 8), name='Cond')

        # Quantize input
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

        x = QActivation(
            quantized_bits(bits=8, integer=1), 
            name='act'
        )(x)

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

        # If bits are allocated for output, rescale + saturate
        if bitsPerOutput > 0 and maxBitsPerOutput > 0:
            latent = keras_floor()(latent * outputMaxIntSize)
            latent = keras_minimum()(latent / outputMaxIntSize, sat_val=outputSaturationValue)

        # Concatenate conditions
        latent = concatenate([latent, cond], axis=1)
        encoder = keras.Model([input_enc, cond], latent, name="encoder")

        # -------------------------------------------------------
        # Decoder Definition
        # -------------------------------------------------------
        input_dec = Input(batch_shape=(batch, 24))
        y = Dense(24)(input_dec)
        y = ReLU()(y)
        y = Dense(64)(y)
        y = ReLU()(y)
        y = Dense(128)(y)
        y = ReLU()(y)
        y = Reshape((4, 4, 8))(y)
        y = Conv2DTranspose(1, (3,3), strides=(2,2), padding='valid')(y)

        # Slice to 8x8
        y = y[:, 0:8, 0:8]
        y = ReLU()(y)
        recon = y
        decoder = keras.Model([input_dec], recon, name="decoder")

        # -------------------------------------------------------
        # Full Autoencoder
        # -------------------------------------------------------
        cae = Model(
            inputs=[input_enc, cond],
            outputs=decoder([encoder([input_enc, cond])]),
            name="cae"
        )

        # -------------------------------------------------------
        # Choose Loss Function
        # -------------------------------------------------------
        if args.loss == 'mse':
            loss_fn = mean_mse_loss
        elif args.loss == 'tele':
            print('Using telescope MSE (8x8) loss')
            loss_fn = telescopeMSE8x8
        else:
            raise ValueError("Unknown loss function specified.")

        # -------------------------------------------------------
        # Optimizer
        # -------------------------------------------------------
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

        # -------------------------------------------------------
        # Learning-Rate Scheduler
        # -------------------------------------------------------
        initial_lr   = args.lr
        total_epochs = args.nepochs

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

        print(f"Training with {args.lr_scheduler} scheduler")

        # -------------------------------------------------------
        # Optionally Continue Training
        # -------------------------------------------------------
        best_val_loss = 1e9
        best_weights_path = os.path.join(sub_dir, 'best-epoch.tf')

        if args.continue_training and os.path.exists(best_weights_path):
            cae.load_weights(best_weights_path)
            print("Continuing training from saved best model...")

        # -------------------------------------------------------
        # Load Data
        # -------------------------------------------------------
        print('Loading Data...')
        train_loader, test_loader, val_loader = load_pre_processed_data(
            args.num_files, batch, m, args
        )
        print('Data Loaded!')

        # Dump training info
        info_txt_path = os.path.join(sub_dir, 'training_info.txt')
        with open(info_txt_path, 'w') as f:
            f.write(f"Training dataset size: {len(train_loader)*args.batchsize}\n")
            f.write(f"Validation dataset size: {len(val_loader)*args.batchsize}\n")
            f.write(f"Test dataset size: {len(test_loader)*args.batchsize}\n")
            f.write("Arguments:\n")
            for arg in vars(args):
                f.write(f"{arg}: {getattr(args, arg)}\n")

        # -------------------------------------------------------
        # If continuing training, load existing logs if present
        # -------------------------------------------------------
        df_csv_path = os.path.join(sub_dir, 'df.csv')
        if args.continue_training and os.path.exists(df_csv_path):
            df_existing = pd.read_csv(df_csv_path)
            loss_dict = {
                'train_loss': df_existing['train_loss'].tolist(),
                'val_loss':   df_existing['val_loss'].tolist()
            }
            start_epoch = len(loss_dict['train_loss']) + 1
        else:
            start_epoch = 1
            loss_dict = {'train_loss': [], 'val_loss': []}

        # -------------------------------------------------------
        # Training Loop
        # -------------------------------------------------------
        for epoch in range(start_epoch, total_epochs + 1):

            # manually set lr each epoch
            new_lr = lr_schedule(epoch)
            tf.keras.backend.set_value(opt.learning_rate, new_lr)

            # Training
            total_loss_train = 0
            for wafers, cond_data in train_loader:
                loss_batch = cae.train_on_batch([wafers, cond_data], wafers)
                total_loss_train += loss_batch
            total_loss_train /= len(train_loader)

            # Validation
            total_loss_val = 0
            for wafers, cond_data in test_loader:
                loss_batch_val = cae.test_on_batch([wafers, cond_data], wafers)
                total_loss_val += loss_batch_val
            total_loss_val /= len(test_loader)

            print(f"Epoch {epoch:03d}, "
                  f"Loss: {total_loss_train:.8f}, "
                  f"ValLoss: {total_loss_val:.8f}")

            # Log
            loss_dict['train_loss'].append(total_loss_train)
            loss_dict['val_loss'].append(total_loss_val)
            df_log = pd.DataFrame.from_dict(loss_dict)

            # Save training curves
            plt.figure(figsize=(10, 6))
            plt.plot(df_log['train_loss'], label='Training Loss')
            plt.plot(df_log['val_loss'],   label='Validation Loss')
            plt.title('Training and Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.yscale('log')
            plt.legend()
            plt.grid(True)
            plot_path = os.path.join(sub_dir, "training_loss_plot.png")
            plt.savefig(plot_path)
            plt.close()

            # Save CSV log
            df_log.to_csv(df_csv_path, index=False)

            # Save best model
            if total_loss_val < best_val_loss:
                print("New Best Model Found!")
                best_val_loss = total_loss_val
                cae.save_weights(os.path.join(sub_dir, 'best-epoch.tf'))
                encoder.save_weights(os.path.join(sub_dir, 'best-encoder-epoch.tf'))
                decoder.save_weights(os.path.join(sub_dir, 'best-decoder-epoch.tf'))

        # -------------------------------------------------------
        # After Training: Save Entire Model
        # -------------------------------------------------------
        save_models(cae, sub_dir, args.mname, isQK=True)


################################################################
# Main: set up a grid search over (lr, batch, optim, lr_sched, nepochs)
################################################################
def main():
    # ----------------------------------------------------------
    # Parse Command-Line Arguments (same as your original code)
    # ----------------------------------------------------------
    p = ArgumentParser()

    # Paths
    p.add_argument('--opath', type=str, required=True)
    p.add_argument('--mpath', type=str, required=False)

    # Model parameters
    p.add_argument('--mname', type=str, required=True)
    p.add_argument('--model_per_eLink', action='store_true')
    p.add_argument('--model_per_bit_config', action='store_true')
    p.add_argument('--alloc_geom', type=str, choices=['old', 'new'], default='old')
    p.add_argument('--specific_m', type=int, required=True)

    # Training parameters
    p.add_argument('--continue_training', action='store_true')
    p.add_argument('--loss', type=str, default='tele')
    p.add_argument('--lr', type=float, required=True)  # will override in grid
    p.add_argument('--nepochs', type=int, required=True)  # override
    p.add_argument('--batchsize', type=int, required=True)  # override
    p.add_argument('--optim', type=str, choices=['adam', 'lion'], default='lion')  # override
    p.add_argument('--lr_scheduler', type=str, choices=['cos', 'cos_warm_restarts'], default='cos_warm_restarts')  # override

    # Dataset parameters
    p.add_argument('--data_path', type=str, required=True)
    p.add_argument('--num_files', type=int, required=True)
    p.add_argument('--train_dataset_size', type=int, default=500000)
    p.add_argument('--val_dataset_size', type=int, default=100000)
    p.add_argument('--test_dataset_size', type=int, default=100000)

    args = p.parse_args()

    # ----------------------------------------------------------
    # Create top-level directory for our grid search
    # ----------------------------------------------------------
    grid_search_dir = os.path.join(args.opath, f'grid_search_eLink_{args.specific_m}')
    os.makedirs(grid_search_dir, exist_ok=True)

    # ----------------------------------------------------------
    # Define the grid for the 5 hyperparameters
    # ----------------------------------------------------------
    lr_values = [5e-4, 1e-4, 5e-5]
    batch_values = [128, 256, 512, 1024]
    optim_values = ['adam', 'lion']
    sched_values = ['cos', 'cos_warm_restarts']
    epoch_values = [50, 100, 200]

    search_space = list(itertools.product(lr_values,
                                         batch_values,
                                         optim_values,
                                         sched_values,
                                         epoch_values))
    # Each element of search_space is a tuple: (lr, batch, optim, scheduler, nepochs)

    # ----------------------------------------------------------
    # Grid search loop
    # ----------------------------------------------------------
    for combo_idx, (lr_, batch_, optim_, sched_, n_epochs_) in enumerate(search_space, start=1):
        print(f"\n=== Grid Search Combo {combo_idx}/{len(search_space)} ===")
        print(f"LR={lr_}, Batch={batch_}, Optim={optim_}, "
              f"Scheduler={sched_}, Epochs={n_epochs_}")

        # Override args
        args.lr = lr_
        args.batchsize = batch_
        args.optim = optim_
        args.lr_scheduler = sched_
        args.nepochs = n_epochs_

        # Create a unique subdirectory for this combo
        subdir_name = (f"lr_{lr_}_bs_{batch_}_opt_{optim_}_"
                       f"sched_{sched_}_ep_{n_epochs_}")
        combo_dir = os.path.join(grid_search_dir, subdir_name)
        os.makedirs(combo_dir, exist_ok=True)

        # Train for this configuration
        train_single_configuration(args, combo_dir)

    print("\n>>> Grid Search Complete! <<<")


if __name__ == "__main__":
    main()
