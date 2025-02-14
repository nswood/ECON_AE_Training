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


class PrintRegularizationLossesPerBatch(tf.keras.callbacks.Callback):
    def on_train_batch_end(self, batch, logs=None):
        regularization_losses = sum(self.model.losses)
        print(f"Batch {batch + 1}: Regularization losses = {regularization_losses}")


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
# LR Scheduler Factory
##############################################################################
def scheduler_factory(cur_hp, initial_lr, max_epochs):
    """
    Returns a Keras LearningRateScheduler callback that adjusts LR each epoch
    based on a chosen scheduler hyperparameter.
    """
    lr_sched = cur_hp.Choice("lr_scheduler", 
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
def build_cae(cur_hp, args):
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
    lr = cur_hp.Float("lr", min_value=1e-5, max_value=1e-2, sampling="log", default=1e-3)
    # loss function
    loss_type = 'tele'
    # optimizer
    optim_type = cur_hp.Choice("optimizer", ["adam", "lion"], default="adam")
    # weight decay
    weight_decay = cur_hp.Float("weight_decay", min_value=1e-6, max_value=1e-2,
                            sampling="log", default=1e-4)
    
    if args.orthogonal_regularization_factor < 0:
        cur_orthogonal_regularization_factor = cur_hp.Float("orthogonal_regularization_factor", min_value=1e-4, max_value=1.0, default=0.01, sampling="log")
        cae = build_cae_model(bitsPerOutput, cur_orthogonal_regularization_factor)

    else:
        cae = build_cae_model(bitsPerOutput, args.orthogonal_regularization_factor)

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
        
        # 7) Log hyperparameters + final metrics to the HParams plugin
        with tf.summary.create_file_writer(trial_log_dir).as_default():
            # 5) Train the model, logging each epoch to TensorBoard
            

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

            print(history.history.keys())  

            # 6) Get the best validation loss
            best_val_loss = min(history.history["val_loss"])

            hp.hparams(cur_hp.values)  # record the used hyperparameters
            tf.summary.scalar("best_val_loss", best_val_loss, step=1)

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
    # Set a fixed seed for reproducibility
    seed_value = 42
    tf.keras.utils.set_random_seed(seed_value)
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
        project_name='cae_project',
        overwrite=False
    )
    if not args.just_write_best_hyperparameters:
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
    log_dir = f'./{args.base_logdir}/eLink_{args.specific_m}'
    if not args.skip_to_final_model:
        tuner.search(
            train_dataset,
            val_dataset,
            max_epochs,
            base_log_dir=log_dir  # used by run_trial
        )
    else:
        print('Skipping to final model training...')
    
    # Retrieve the best hyperparams
    best_hp = tuner.get_best_hyperparameters(num_trials=args.num_trials)[0]
    print("Best Hyperparameters:", best_hp.values)
    print('Best loss', tuner.oracle.get_best_trials(1)[0].score)
    lowest_loss = tuner.oracle.get_best_trials(1)[0].score

    # Build the best model
    best_model = tuner.hypermodel.build(best_hp)

    # -------------------------------------------------
    # Optional: Re-train the best model in a 'final' run
    # and log that training to TensorBoard as well.
    # -------------------------------------------------
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
        max_epochs = 100
        performance_file = os.path.join(final_log_dir, 'performance_records.csv')
        if os.path.exists(performance_file):
            existing_records = pd.read_csv(performance_file)
            start_seed = len(existing_records)
            mode = 'a'  # append mode
        else:
            start_seed = 0
            mode = 'w'  # write mode

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
                    best_val_loss = seed_val_loss
                    best_model_weights = best_model.get_weights()

                # Write the current seed's performance to the CSV file
                writer.writerow([seed, seed_val_loss])
                f.flush()

            # Save the best model found from the search
            def save_model(model, dir_name, model_name):
                output_model_dir = os.path.join(tuner_dir, dir_name)
                os.makedirs(output_model_dir, exist_ok=True)
                save_models(model, output_model_dir, model_name, isQK=True)
                print(f"{model_name} saved to: {output_model_dir}")

            save_model(best_model, f'best_model_eLink_{args.specific_m}', 'best_model')

            # Load larger dataset to train final model on larger set
            print('Loading Larger Final Data...')
            train_dataset, test_dataset, val_dataset = load_pre_processed_data_for_hyperband(
                args.num_files, args.specific_m, args, dataset_limit=500_000
            )

            train_dataset = train_dataset.map(format_for_autoencoder)
            val_dataset = val_dataset.map(format_for_autoencoder)

            # Rebuild the best model with the best hyperparameters and best initial seed
            best_model = tuner.hypermodel.build(best_hp)
            best_model.set_weights(best_model_weights)

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
            save_model(best_model, f'final_best_model_eLink_{args.specific_m}', 'final_best_model')

            # Save a file showing the val_loss for the seed training and the larger dataset training
            with open(os.path.join(final_log_dir, 'final_val_loss.csv'), 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['Training Phase', 'Validation Loss'])
                writer.writerow(['Seed Search', best_val_loss])
                writer.writerow(['Larger Dataset Training', final_val_loss])
                

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
    parser.add_argument('--orthogonal_regularization_factor', type=float, default=0.0)
    parser.add_argument('--base_logdir', type=str, default='./logs')
    parser.add_argument('--just_write_best_hyperparameters', action='store_true')
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
        HP_ORTHOGONAL_REGULARIZATION_FACTOR = hp.HParam('orthogonal_regularization_factor', hp.RealInterval(1e-4, 1.0))

    # Set up a top-level logging directory for all eLink_{args.specific_m} runs
    top_level_log_dir = f'{args.base_logdir}/eLink_{args.specific_m}'
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
