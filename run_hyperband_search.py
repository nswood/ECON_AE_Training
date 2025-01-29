import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
import keras_tuner
from keras_tuner import Hyperband
from tensorflow.keras.callbacks import TensorBoard
from tensorboard.plugins.hparams import api as hp

# If you're using GPU:
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

# Keras / QKeras
from tensorflow import keras
from keras.layers import (Layer, Input, Flatten, Dense, ReLU, Reshape,
                          Conv2DTranspose, concatenate)
from keras.models import Model
import qkeras
from qkeras import QActivation, QConv2D, QDense, quantized_bits
from utils.telescope import telescopeMSE8x8
from utils.utils import *

##############################################################################
# Custom Layers
##############################################################################
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

##############################################################################
# LR Scheduler Factory
##############################################################################
def scheduler_factory(hp, initial_lr, max_epochs):
    """
    Returns a Keras LearningRateScheduler callback that adjusts LR each epoch
    based on a chosen scheduler hyperparameter.
    """
    lr_sched = hp.Choice("lr_scheduler", 
                         ["cos", "cos_warm_restarts", "step_decay", "exp_decay"],
                         default="cos")

    if lr_sched == "cos":
        # Simple cosine annealing
        def cos_anneal(epoch):
            cos_inner = np.pi * (epoch % max_epochs) / max_epochs
            return initial_lr / 2 * (np.cos(cos_inner) + 1.0)
        return keras.callbacks.LearningRateScheduler(cos_anneal, verbose=0)

    elif lr_sched == "cos_warm_restarts":
        # Example: smaller restarts
        def cos_warm(epoch):
            T = max_epochs // 3 if max_epochs > 3 else 1
            cycle = epoch % T
            cos_inner = np.pi * cycle / T
            return initial_lr / 2 * (np.cos(cos_inner) + 1.0)
        return keras.callbacks.LearningRateScheduler(cos_warm, verbose=0)

    elif lr_sched == "step_decay":
        # e.g. reduce LR by 10x every 10 epochs
        def step_dec(epoch):
            drop_every = 10
            drop_factor = 0.1
            return initial_lr * (drop_factor ** (epoch // drop_every))
        return keras.callbacks.LearningRateScheduler(step_dec, verbose=0)

    elif lr_sched == "exp_decay":
        # e.g. exponential decay
        def exp_dec(epoch):
            k = 0.96
            return initial_lr * (k ** epoch)
        return keras.callbacks.LearningRateScheduler(exp_dec, verbose=0)


##############################################################################
# Build & Compile the CAE
##############################################################################
def build_cae(hp, args):
    """
    Build and compile the CAE model using hyperparameters from Keras Tuner,
    plus a *fixed* bitsPerOutput from args.
    """

    # -----------------------------------------------------------
    # 1) Determine bitsPerOutput from your existing logic
    # -----------------------------------------------------------
    # only support for model_per_eLink:
    bitsPerOutputLink = [0,1,3,5,7,9,9,9,9,9,9,9,9,9,9]
    eLinks = args.specific_m
    bitsPerOutput = bitsPerOutputLink[eLinks]

    # -----------------------------------------------------------
    # 2) Hyperparameters from Keras Tuner
    # -----------------------------------------------------------
    # learning rate
    lr = hp.Float("lr", min_value=1e-5, max_value=1e-2, sampling="log", default=1e-3)
    # loss function
    loss_type = 'tele'
    # optimizer
    optim_type = hp.Choice("optimizer", ["adam", "lion"], default="adam")
    # weight decay
    weight_decay = hp.Float("weight_decay", min_value=1e-6, max_value=1e-2,
                            sampling="log", default=1e-4)
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

    # -----------------------------------------------------------
    # 6) Compile with chosen loss & optimizer
    # -----------------------------------------------------------
    if loss_type == "mse":
        loss_fn = mean_mse_loss
    else:
        loss_fn = telescopeMSE8x8

    if optim_type == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr, weight_decay=weight_decay)
    else:
        # Lion requires TF >= 2.12 or a separate add-on
        optimizer = tf.keras.optimizers.Lion(learning_rate=lr, weight_decay=weight_decay)

    cae.compile(optimizer=optimizer, loss=loss_fn)
    return cae

##############################################################################
# A Custom Hyperband Tuner so we can tune batch_size via run_trial
##############################################################################
class MyHyperband(Hyperband):
    """
    Subclass Hyperband so we can override run_trial and dynamically set
    batch_size (among other parameters) in model.fit(...), and also log
    each trial's progress to TensorBoard.
    """
    def run_trial(self, trial, train_dataset, val_dataset, max_epochs, base_log_dir):
        hp = trial.hyperparameters

        # 1) Build the model for this trial
        model = self.hypermodel.build(hp)
    
        # 2) Figure out batch_size from HP
        batch_size = hp.Choice("batch_size", [64, 128, 256, 512, 1024], default=256)
        init_lr = hp.get("lr")

        # 3) Create a subdirectory for this trial's logs
        trial_log_dir = os.path.join(base_log_dir, f"trial_{trial.trial_id}")
        os.makedirs(trial_log_dir, exist_ok=True)

        # 4) TensorBoard callback for per-epoch logging
        tb_callback = tf.keras.callbacks.TensorBoard(
            log_dir=trial_log_dir,
            update_freq="epoch"
        )

        # 5) Train the model, logging each epoch to TensorBoard
        history = model.fit(
            train_dataset.batch(batch_size),
            validation_data=val_dataset.batch(batch_size),
            epochs=max_epochs,
            verbose=0,
            callbacks=[
                tb_callback,
                scheduler_factory(hp, init_lr, max_epochs),
            ]
        )

        # 6) Get the best validation loss
        best_val_loss = min(history.history["val_loss"])

        # 7) Log hyperparameters + final metrics to the HParams plugin
        with tf.summary.create_file_writer(trial_log_dir).as_default():
            hp.hparams(hp.values)  # record the used hyperparameters
            tf.summary.scalar("best_val_loss", best_val_loss, step=max_epochs)

        # 8) Update Keras Tuner with the best validation loss
        self.oracle.update_trial(trial.trial_id, {"val_loss": best_val_loss})


##############################################################################
# Utility for formatting the data for your model
##############################################################################
def format_for_autoencoder(wafer, cond):
    # cond shape => (batch, 8)
    # wafer shape => (batch, 8, 8, 1)
    return (wafer, cond), wafer  # x=(wafer, cond), y=wafer

##############################################################################
# Main Tuning Function
##############################################################################
def run_hyperband(args):
    # Create output directory
    tuner_dir = os.path.join(args.opath, f'hyperband_search_{args.specific_m}')
    os.makedirs(tuner_dir, exist_ok=True)

    max_epochs = 50  # The largest bracket for Hyperband

    # Create our custom tuner
    tuner = MyHyperband(
        hypermodel=lambda hp: build_cae(hp, args),
        objective='val_loss',
        max_epochs=max_epochs,
        factor=3,
        directory=tuner_dir,
        project_name='cae_project'
    )

    print('Loading Data...')
    train_dataset, test_dataset, val_dataset = load_pre_processed_data_for_hyperband(
        args.num_files, args.specific_m, args
    )

    # Format the datasets for your autoencoder model
    train_dataset = train_dataset.map(format_for_autoencoder)
    val_dataset = val_dataset.map(format_for_autoencoder)

    print('Sample shapes from training dataset:')
    for batch in train_dataset.take(1):
        x_batch, y_batch = batch
        wafer_input, cond_input = x_batch
        print(f"Wafer shape: {wafer_input.shape}")   # e.g. (batch_size, 8, 8, 1)
        print(f"Cond shape: {cond_input.shape}")     # e.g. (batch_size, 8)
        print(f"Target shape: {y_batch.shape}")      # e.g. (batch_size, 8, 8, 1)
    print('Data Loaded!')

    # -------------------------------------------------
    # Run the Hyperband search
    # Note: We pass the 'base_log_dir' so each trial
    # can log to its own subdirectory for TensorBoard.
    # -------------------------------------------------
    log_dir = f'./logs/eLink_{args.specific_m}'
    tuner.search(
        train_dataset,
        val_dataset,
        max_epochs,
        base_log_dir=log_dir  # used by run_trial
    )

    # Retrieve the best hyperparams
    best_hp = tuner.get_best_hyperparameters(num_trials=50)[0]
    print("Best Hyperparameters:", best_hp.values)

    # Build the best model
    best_model = tuner.hypermodel.build(best_hp)

    # -------------------------------------------------
    # Optional: Re-train the best model in a 'final' run
    # and log that training to TensorBoard as well.
    # -------------------------------------------------
    final_log_dir = os.path.join(log_dir, 'final')
    os.makedirs(final_log_dir, exist_ok=True)

    init_lr = best_hp.get("lr")
    final_cb = [
        scheduler_factory(best_hp, init_lr, max_epochs),
        TensorBoard(log_dir=final_log_dir, update_freq="epoch")
    ]
    batch_size = best_hp.get("batch_size", 32)  # default if not set
    history = best_model.fit(
        train_dataset.batch(batch_size),
        validation_data=val_dataset.batch(batch_size),
        epochs=max_epochs,
        callbacks=final_cb,
        verbose=1
    )

    best_val_loss = np.min(history.history["val_loss"])
    print(f"Final model after re-training has best_val_loss = {best_val_loss:.4f}")

    # Save the final best model
    output_model_dir = os.path.join(tuner_dir, f'best_model_eLink_{args.specific_m}')
    save_models(best_model, output_model_dir, 'best_model', isQK=True)
    print(f"Best model saved to: {output_model_dir}")


##############################################################################
# Main
##############################################################################
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--opath', type=str, required=True, help="Output path")
    parser.add_argument('--mname', type=str, required=True, help="Model name")
    parser.add_argument('--model_per_eLink', action='store_true')
    parser.add_argument('--model_per_bit_config', action='store_true')
    parser.add_argument('--specific_m', type=int, required=True)
    parser.add_argument('--num_files', type=int, default=1)
    parser.add_argument('--data_path', type=str, default='data')
    args = parser.parse_args()

    # ---------------------------------------------------------------
    # Define HParam ranges (for the HP dashboard in TensorBoard)
    # ---------------------------------------------------------------
    HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'lion']))
    HP_LR = hp.HParam('lr', hp.RealInterval(1e-5, 1e-2))
    HP_LR_SCHED = hp.HParam('lr_scheduler', hp.Discrete(['cos', 'cos_warm_restarts', 'step_decay', 'exp_decay']))
    HP_WEIGHT_DECAY = hp.HParam('weight_decay', hp.RealInterval(1e-6, 1e-2))
    HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([64, 128, 256, 512, 1024]))

    # Set up a top-level logging directory for all eLink_{args.specific_m} runs
    top_level_log_dir = f'./logs/eLink_{args.specific_m}'
    os.makedirs(top_level_log_dir, exist_ok=True)

    # Log the hyperparameter configuration once so the HParams plugin knows
    with tf.summary.create_file_writer(top_level_log_dir).as_default():
        hp.hparams_config(
            hparams=[HP_OPTIMIZER, HP_LR, HP_LR_SCHED, HP_WEIGHT_DECAY, HP_BATCH_SIZE],
            metrics=[hp.Metric('val_loss', display_name='Validation Loss')],
        )

    run_hyperband(args)

if __name__ == "__main__":
    main()
