import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
import keras_tuner
from keras_tuner import Hyperband
from tensorflow.keras.callbacks import TensorBoard
from tensorboard.plugins.hparams import api as hp
import argparse
import csv

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
from keras.layers import (Input, Flatten, Dense, ReLU, Reshape, Conv2DTranspose)
from keras.models import Model
from tensorflow.keras import regularizers

import qkeras
from qkeras import QActivation, QConv2D, QDense, quantized_bits

# Project utils
from utils.telescope import telescopeMSE8x8
from utils.utils import *  # expects mean_mse_loss, save_models, data loaders, etc.
from utils.fix_preprocess_CMSSW import *


##############################################################################
# Model Saving Utilities
##############################################################################

def save_AE_models(autoencoder,model_dir,  name, isQK=False):
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
    graph.write_frozen_dummy_enc(encoder,  f"encoder_{name}.pb",       logdir=model_dir)
    graph.write_frozen_dummy_enc(encoder,  f"encoder_{name}.pb.ascii", logdir=model_dir, asText=True)
    graph.write_frozen_graph_dec(decoder,  f"decoder_{name}.pb",       logdir=model_dir)
    graph.write_frozen_graph_dec(decoder,  f"decoder_{name}.pb.ascii", logdir=model_dir, asText=True)

    # Plot weights for debugging/inspection
    graph.plot_weights(autoencoder, outdir=model_dir)
    graph.plot_weights(encoder,     outdir=model_dir)
    graph.plot_weights(decoder,     outdir=model_dir)

##############################################################################
# LR Scheduler Factory
##############################################################################
def scheduler_factory(cur_hp, initial_lr, max_epochs):
    """
    Returns a Keras LearningRateScheduler callback that adjusts LR each epoch
    based on a chosen scheduler hyperparameter.
    """
    lr_sched = cur_hp.Choice(
        "lr_scheduler",
        ["cos", "cos_warm_restarts", "step_decay", "exp_decay"],
        default="cos"
    )

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



def build_ae_model(bitsPerOutput, orthogonal_regularization_factor = 0):
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

        encoder = keras.Model(input_enc, latent, name="encoder")

        # -----------------------------------------------------------
        # 5) Build Decoder
        # -----------------------------------------------------------
        input_dec = Input(shape=(16,))
        y = Dense(16,kernel_regularizer=regularizer)(input_dec)
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
        ae = Model(
            inputs=input_enc,
            outputs=decoder(encoder(input_enc)),
            name="ae"
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
        encoder = keras.Model(input_enc, latent, name="encoder")

        # -----------------------------------------------------------
        # 5) Build Decoder
        # -----------------------------------------------------------
        input_dec = Input(shape=(16,))
        y = Dense(16)(input_dec)
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
        ae = Model(
            inputs=input_enc,
            outputs=decoder(encoder(input_enc)),
            name="cae"
        )
    return ae

##############################################################################
# Build & Compile the AE
##############################################################################
def build_ae(cur_hp, args, loss_type='tele'):
    """
    Build and compile the unconditional AE model using hyperparameters from Keras Tuner,
    plus a bitsPerOutput derived from eLinks if desired.
    """
    # Map eLinks -> bitsPerOutput (kept from your original logic)
    bitsPerOutputLink = [0, 1, 3, 5, 7, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9]
    eLinks = args.specific_m
    bitsPerOutput = bitsPerOutputLink[eLinks]

    # Hyperparameters from Keras Tuner
    lr = cur_hp.Float("lr", min_value=1e-5, max_value=1e-2, sampling="log", default=1e-3)
    optim_type = cur_hp.Choice("optimizer", ["adam", "lion"], default="adam")
    weight_decay = cur_hp.Float(
        "weight_decay", min_value=1e-6, max_value=1e-2, sampling="log", default=1e-4
    )

    # Build model with tuned/arg regularization
    if args.orthogonal_regularization_factor < 0:
        cur_orthogonal_regularization_factor = cur_hp.Float(
            "orthogonal_regularization_factor", min_value=1e-4, max_value=1.0,
            default=0.01, sampling="log"
        )
        ae = build_ae_model(bitsPerOutput, cur_orthogonal_regularization_factor)
    else:
        ae = build_ae_model(bitsPerOutput, args.orthogonal_regularization_factor)

    # Loss
    loss_fn = mean_mse_loss if loss_type == "mse" else telescopeMSE8x8

    # Optimizer (Lion requires TF >= 2.12 or addon)
    if optim_type == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr, weight_decay=weight_decay)
    else:
        optimizer = tf.keras.optimizers.Lion(learning_rate=lr, weight_decay=weight_decay)

    ae.compile(optimizer=optimizer, loss=loss_fn)
    return ae


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
        cur_hp = trial.hyperparameters

        # 1) Build the model for this trial
        model = self.hypermodel.build(cur_hp)

        # 2) Figure out batch_size from HP
        batch_size = cur_hp.Choice("batch_size", [64, 128, 256, 512, 1024], default=256)
        init_lr = cur_hp.get("lr")

        # 3) Create a subdirectory for this trial's logs
        trial_log_dir = os.path.join(base_log_dir, f"trial_{trial.trial_id}")
        os.makedirs(trial_log_dir, exist_ok=True)

        # 4) TensorBoard callback for per-epoch logging
        tb_callback = tf.keras.callbacks.TensorBoard(
            log_dir=trial_log_dir,
            update_freq="epoch"
        )

        # 5) Train the model, logging each epoch to TensorBoard
        with tf.summary.create_file_writer(trial_log_dir).as_default():
            history = model.fit(
                train_dataset.batch(batch_size),
                validation_data=val_dataset.batch(batch_size),
                epochs=max_epochs,
                verbose=0,
                callbacks=[
                    tb_callback,
                    scheduler_factory(cur_hp, init_lr, max_epochs),
                ]
            )

            # 6) Get the best validation loss
            best_val_loss = min(history.history["val_loss"])

            # 7) Log hyperparameters + final metrics to the HParams plugin
            hp.hparams(cur_hp.values)  # record the used hyperparameters
            tf.summary.scalar("best_val_loss", best_val_loss, step=1)

        # 8) Update Keras Tuner with the best validation loss
        self.oracle.update_trial(trial.trial_id, {"val_loss": best_val_loss})


##############################################################################
# Utility for formatting the data for your model
##############################################################################
def format_for_autoencoder(wafer, _cond_unused):
    """
    Unconditional AE formatting: drop conditioning; x = wafer, y = wafer.
    """
    return wafer, wafer


##############################################################################
# Main Tuning Function
##############################################################################
def run_hyperband(args):
    # Set a fixed seed for reproducibility

    
    seed_value = 42
    tf.keras.utils.set_random_seed(seed_value)

    # Create output directory
    tuner_dir = os.path.join(args.opath, f'hyperband_search_{args.specific_m}')
    os.makedirs(tuner_dir, exist_ok=True)

    max_epochs = 2 if args.test_run else 50  # Largest bracket for Hyperband

    # Create our custom tuner (now uses build_ae)
    tuner = MyHyperband(
        hypermodel=lambda hp: build_ae(hp, args),
        objective='val_loss',
        max_epochs=max_epochs,
        factor=3,
        directory=tuner_dir,
        project_name='cae_hyperband_base_dir',
        overwrite=False
    )

    if not args.just_write_best_hyperparameters:
        if args.test_run:
            args.num_files = 1
        print('Loading Data...')
        train_dataset, test_dataset, val_dataset = load_pre_processed_data_for_hyperband(
            args.num_files, args.specific_m, args
        )

        # Format the datasets for your autoencoder model (drop cond)
        train_dataset = train_dataset.map(format_for_autoencoder)
        val_dataset = val_dataset.map(format_for_autoencoder)

        print('Sample shapes from training dataset:')
        for batch in train_dataset.take(1):
            x_batch, y_batch = batch
            print(f"Wafer shape: {x_batch.shape}")    # e.g. (batch_size, 8, 8, 1)
            print(f"Target shape: {y_batch.shape}")   # e.g. (batch_size, 8, 8, 1)
        print('Data Loaded!')

    # -------------------------------------------------
    # Run the Hyperband search
    # -------------------------------------------------
    log_dir = f'{args.opath}/hyperband_search_{args.specific_m}/log'
    if not args.skip_to_final_model:
        tuner.search(
            train_dataset,
            val_dataset,
            max_epochs,
            base_log_dir=log_dir  # used by run_trial
        )
    else:
        print('Skipping to final model training...')

    if args.test_run:
        args.num_trials = 2

    # Retrieve the best hyperparams
    best_hp = tuner.get_best_hyperparameters(num_trials=args.num_trials)[0]
    print("Best Hyperparameters:", best_hp.values)
    print('Best loss', tuner.oracle.get_best_trials(1)[0].score)
    lowest_loss = tuner.oracle.get_best_trials(1)[0].score

    # Build the best model
    best_model = tuner.hypermodel.build(best_hp)

    final_log_dir = os.path.join(log_dir, 'final')
    os.makedirs(final_log_dir, exist_ok=True)

    best_val_loss = float('inf')
    best_model_weights = None

    # Save hyperparameters
    with open(os.path.join(final_log_dir, 'best_hyperparameters.csv'), 'w') as f:
        writer = csv.writer(f)
        for key, value in best_hp.values.items():
            writer.writerow([key, value])
        writer.writerow(['best_val_loss', lowest_loss])

    if not args.just_write_best_hyperparameters:
        # Train over args.num_seeds initial seeds
        performance_records = []
        max_epochs = 2 if args.test_run else 100
        args.num_seeds = 2 if args.test_run else args.num_seeds

        performance_file = os.path.join(final_log_dir, 'performance_records.csv')
        if os.path.exists(performance_file):
            existing_records = pd.read_csv(performance_file)
            start_seed = len(existing_records)
            mode = 'a'  # append mode
        else:
            start_seed = 0
            mode = 'w'  # write mode
        if start_seed > args.num_seeds:
            print(f"Already trained {start_seed} seeds, skipping further training.")
            best_seed = existing_records.loc[existing_records['Validation Loss'].idxmin(), 'Seed']
            best_seed = int(best_seed)
            best_val_loss = existing_records['Validation Loss'].min()
            best_val_loss = float(best_val_loss)
            init_lr = best_hp.get("lr")
            batch_size = best_hp.get("batch_size")
            
        else:
            with open(performance_file, mode) as f:
                writer = csv.writer(f)
                if start_seed == 0:
                    writer.writerow(['Seed', 'Validation Loss'])
                
                for base_seed in range(start_seed, start_seed + args.num_seeds):
                    seed = base_seed
                    # Rebuild the model for each seed to ensure re-initialization
                    best_model = tuner.hypermodel.build(best_hp)
                    print(f"Training with seed {seed}...")
                    tf.keras.utils.set_random_seed(seed)

                    init_lr = best_hp.get("lr")
                    final_cb = [
                        scheduler_factory(best_hp, init_lr, max_epochs),
                        TensorBoard(log_dir=os.path.join(final_log_dir, f'seed_{seed}'), update_freq="epoch")
                    ]
                    batch_size = best_hp.get("batch_size")
                    history = best_model.fit(
                        train_dataset.batch(batch_size),
                        validation_data=val_dataset.batch(batch_size),
                        epochs=max_epochs,
                        callbacks=final_cb,
                        verbose=1
                    )

                    seed_val_loss = np.min(history.history["val_loss"])
                    performance_records.append([seed, seed_val_loss])
                    print(f"Seed {seed} model has val_loss = {seed_val_loss:.4f}")

                    # Save the best model weights
                    if seed_val_loss < best_val_loss:
                        best_seed = seed
                        best_val_loss = seed_val_loss
                        best_model_weights = best_model.get_weights()

                    # Write the current seed's performance to the CSV file
                    writer.writerow([seed, seed_val_loss])
                    f.flush()
            
        # Save the best model from the seed sweep
        def save_model(model, dir_name, model_name):
            output_model_dir = os.path.join(tuner_dir, dir_name)
            os.makedirs(output_model_dir, exist_ok=True)
            save_AE_models(model, output_model_dir, model_name, isQK=True)
            print(f"{model_name} saved to: {output_model_dir}")

        # Write best seed
        with open(os.path.join(final_log_dir, 'best_hyperparameters.csv'), 'a') as f2:
            writer2 = csv.writer(f2)
            writer2.writerow(['best_seed', best_seed])
            writer2.writerow(['best_seed_val_loss', best_val_loss])

        save_model(best_model, f'best_model_eLink_{args.specific_m}_post_seed_variation', 'model')

        save_CMSSW_compatible_AEmodel(
            mname=args.mname,
            mpath=os.path.join(tuner_dir, f'best_model_eLink_{args.specific_m}_post_seed_variation'),
            outdir=os.path.join(tuner_dir, f'best_model_eLink_{args.specific_m}_post_seed_variation_for_CMSSW'),
            eLinks=args.specific_m,
            alloc_geom=None
        )

        # Load larger dataset to train final model on larger set
        print('Loading Larger Final Data...')
        if args.test_run:
            train_dataset, test_dataset, val_dataset = load_pre_processed_data_for_hyperband(
                args.num_files, args.specific_m, args, dataset_limit=25_000
            )
        else:
            train_dataset, test_dataset, val_dataset = load_pre_processed_data_for_hyperband(
                args.num_files, args.specific_m, args, dataset_limit=500_000
            )

        train_dataset = train_dataset.map(format_for_autoencoder)
        val_dataset = val_dataset.map(format_for_autoencoder)

        tf.keras.utils.set_random_seed(best_seed)
        # Rebuild the best model with the best hyperparameters and best initial seed
        best_model = tuner.hypermodel.build(best_hp)
        # best_model.set_weights(best_model_weights)

        # Train the best model on the larger dataset
        final_cb = [
            scheduler_factory(best_hp, init_lr, max_epochs),
            TensorBoard(log_dir=os.path.join(final_log_dir, 'final_training'), update_freq="epoch")
        ]
        history = best_model.fit(
            train_dataset.batch(batch_size),
            validation_data=val_dataset.batch(batch_size),
            epochs=max_epochs,
            callbacks=final_cb,
            verbose=1
        )

        final_val_loss = np.min(history.history["val_loss"])
        print(f"Final model has val_loss = {final_val_loss:.4f}")

        # Save the final best model trained on the larger dataset
        save_model(best_model, f'best_model_eLink_{args.specific_m}_post_seed_variation_larger_dataset', 'model')

        # Save a file showing the val_loss for the seed training and the larger dataset training
        with open(os.path.join(final_log_dir, 'final_val_loss.csv'), 'w') as f3:
            writer3 = csv.writer(f3)
            writer3.writerow(['Training Phase', 'Validation Loss'])
            writer3.writerow(['Seed Search', best_val_loss])
            writer3.writerow(['Larger Dataset Training', final_val_loss])

        # Process best model into CMSSW
        save_CMSSW_compatible_AEmodel(
            mname=args.mname,
            mpath=os.path.join(tuner_dir, f'best_model_eLink_{args.specific_m}_post_seed_variation_larger_dataset'),
            outdir=os.path.join(tuner_dir, f'best_model_eLink_{args.specific_m}_post_seed_variation_larger_dataset_for_CMSSW'),
            eLinks=args.specific_m,
            alloc_geom=None
            )
    if args.just_save_best_model:
        save_CMSSW_compatible_AEmodel(
                mname=args.mname,
                mpath=os.path.join(tuner_dir, f'best_model_eLink_{args.specific_m}_post_seed_variation_larger_dataset'),
                outdir=os.path.join(tuner_dir, f'best_model_eLink_{args.specific_m}_post_seed_variation_larger_dataset_for_CMSSW'),
                eLinks=args.specific_m,
                alloc_geom=None
                )

##############################################################################
# Main
##############################################################################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--opath', type=str, required=True, help="Output path")
    parser.add_argument('--mname', type=str, required=True, help="Model name")
    parser.add_argument('--model_per_eLink', action='store_true')
    parser.add_argument('--model_per_bit_config', action='store_true')
    parser.add_argument('--specific_m', type=int, required=True)
    parser.add_argument('--num_files', type=int, default=1)
    parser.add_argument('--data_path', type=str, default='data')
    parser.add_argument('--num_trials', type=int, default=50)
    parser.add_argument('--skip_to_final_model', action='store_true')
    parser.add_argument('--test_run', action='store_true')
    parser.add_argument('--orthogonal_regularization_factor', type=float, default=0.0)
    parser.add_argument('--just_write_best_hyperparameters', action='store_true')
    parser.add_argument('--just_save_best_model', action='store_true')
    parser.add_argument('--num_seeds', type=int, default=20)

    args = parser.parse_args()

    # ---------------------------------------------------------------
    # Define HParam ranges (for the HP dashboard in TensorBoard)
    # ---------------------------------------------------------------
    HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'lion']))
    HP_LR = hp.HParam('lr', hp.RealInterval(1e-5, 1e-2))
    HP_LR_SCHED = hp.HParam('lr_scheduler', hp.Discrete(['cos', 'cos_warm_restarts', 'step_decay', 'exp_decay']))
    HP_WEIGHT_DECAY = hp.HParam('weight_decay', hp.RealInterval(1e-6, 1e-2))
    HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([64, 128, 256, 512, 1024]))
    if args.orthogonal_regularization_factor < 0:
        HP_ORTHOGONAL_REGULARIZATION_FACTOR = hp.HParam(
            'orthogonal_regularization_factor', hp.RealInterval(1e-4, 1.0)
        )

    # Set up a top-level logging directory for all eLink_{args.specific_m} runs
    top_level_log_dir = f'{args.opath}/hyperband_search_{args.specific_m}/log'
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
