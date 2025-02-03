# ECON_QAE_Training

## Overview
This repository provides code for training and evaluating Quantized Autoencoders (QAE) as part of the ECON project. The code is organized into various scripts for data processing, model training, and evaluation.

## Setup
To set up the environment, create and activate the Conda environment using the provided YAML file:

```bash
conda env create -f environment.yml
conda activate econ_qae
```

## CAE Description
The Conditional Autoencoder (CAE) consists of a quantized encoder and an unquantized decoder, with additional conditioning in the latent space for known wafer information. Specifically, for HGCAL wafer encoding, the following conditional variables are used:
- eta
- waferu
- waferv
- wafertype (one-hot encoded into 3 possible types)
- sumCALQ
- layers

Altogether, these 8 conditional variables are concatenated with a 16D latent code, resulting in a 24D input to the decoder.

- Training is performed via `train_CAE_simon_data.py`.
- CMSSW integration is handled by `preprocess_CMSSW.py`, which slightly modifies how conditioning is applied (without affecting model performance) to ensure CMSSW compatibility.

## Generating the Dataset
Use the `process_data.py` script to generate or preprocess the dataset. Below is an example command:

```bash
python process_data.py --opath test_data_saving --num_files 2 --model_per_eLink --biased 0.90 --save_every_n_files 1 --alloc_geom old --use_local --seed 12345
```

Arguments:
- `--opath`: Output directory for saved data.
- `--num_files`: Number of ntuples to preprocess.
- `--model_per_eLink`: Trains a unique CAE per possible eLink allocation.
- `--model_per_bit_config`: Trains a unique CAE per possible bit allocation.
- `--biased`: Resamples the dataset so that n% of the data is signal and (1-n)% is background (specify n as a float).
- `--save_every_n_files`: Number of ntuples to combine per preprocessed output file.
- `--alloc_geom`: The allocation geometry (old, new).
- `--use_local`: If passed, read .root files from local directory (for CMU Rogue01 GPU Cluster only). If not passed, it uses XRootD to get the data from Tier 3.
- `--seed`: If provided, enforces a fixed random seed for consistent shuffling and splitting (reproducible train/test splits).

## Training the Model
Use the `train_ECON_AE_CAE.py` script to train the model. The `train_ECON_AE_CAE.py` script automatically runs `preprocess_CMSSW.py` which generates the necessary files to run the trained CAE in CMSSW. Below is an example command:

```bash
python train_ECON_AE_CAE.py --opath test_new_run --mname test --model_per_eLink --alloc_geom old --data_path test_data_saving --loss tele --optim lion --lr 1e-4 --lr_sched cos --train_dataset_size 2000 --test_dataset_size 1000 --val_dataset_size 1000 --batchsize 128 --num_files 1 --nepochs 10 --seed 12345
```

Arguments:
- `--opath`: Output directory for the training run.
- `--mname`: Model name.
- `--model_per_eLink`: Trains a unique CAE per possible eLink allocation.
- `--model_per_bit_config`: Trains a unique CAE per possible bit allocation.
- `--alloc_geom`: Allocation geometry (old, new).
- `--data_path`: Path to the preprocessed dataset.
- `--loss`: Loss function (tele, mse).
- `--optim`: Optimizer (lion, adam).
- `--lr`: Learning rate.
- `--lr_sched`: Learning rate scheduler (cos, cos_warm_restarts).
- `--train_dataset_size`: Number of samples in the training dataset.
- `--test_dataset_size`: Number of samples in the test dataset.
- `--val_dataset_size`: Number of samples in the validation dataset.
- `--batchsize`: Training batch size.
- `--num_files`: Number of preprocessed files to use.
- `--nepochs`: Number of training epochs.
- `--seed`: If passed, fixes the random seed for weight initialization, data ordering, etc., to make results reproducible across machines.

## File Descriptions
- `gen_latent_samples.py`: Generates latent samples for analysis.
- `process_data.py`: Processes raw data and creates datasets.
- `train_ECON_AE_CAE.py`: Trains the ECON Conditional Autoencoder models.
- `preprocess_CMSSW.py`: Preprocesses CAE models for CMSSW.
- `utils/graph.py`: Utility functions for graph operations.
- `utils/utils.py`: General utility functions.
- `utils/telescope.py`: Telescope loss function.
- `utils/files.py`: File I/O helper functions.

## Additional Information
For more details on each script and its usage, please refer to inline comments and docstrings within the code files. If you encounter any issues, feel free to open an issue or submit a pull request.
