# ECON_QAE_Training

## Overview
This repository contains code for training and evaluating Quantized Autoencoders (QAE) for the ECON project. The code is organized into several scripts and utility files to facilitate data processing, model training, and evaluation.

## Setup
To set up the environment, use the provided YAML file to create a conda environment:

```bash
conda env create -f environment.yml
conda activate econ_qae
```

## CAE Description

The Conditional Autoencoder (CAE) is a quantized encoder & unquantized decoder with conditioning in the latent space for known information. For the HGCAL wafer encoding we have the following conditional information: eta, waferu, waferv, wafertype, sumCALQ, layers. We hot-encode wafertype as there are three possible wafertypes. This gives 8 total conditional variables. In combination with the 16D latent space, this means the decoder takes a 24D latent vector to decode the wafer.

Training the model is done through the train_CAE_simon_data.py file. To process data for CMSSW, you must use the CMSSW processing file dev_CMSSW.py. The processing modifies the model architecture so that CMSSW apply conditioning without producing errors. There is no change of this for model performance, just how we feed it into CMSSW.  
 
## Example training CAE model:
To train the model:

```bash
python train_CAE_simon_data.py --mname AE  --batchsize 4000 --lr 1e-4 --nepochs 1000 --opath Tele_CAE_biased_90 --optim lion --loss tele --biased --b_percent 0.90
```

To process the model: 

```bash
python dev_preCMSSW.py --mname AE --mpath Tele_CAE_biased_90
```

## Generating the Dataset
To generate the dataset, use the `process_data.py` script. Below is an example command:

```bash
python process_data.py --opath test_data_saving --num_files 2 --model_per_eLink --biased 0.90 --save_every_n_files 1 --alloc_geom old
```

- `--opath`: Output directory for data saving.
- `--num_files`: Number of ntuples to load.
- `--model_per_eLink`: Trains a unique CAE per possible eLink allocation.
- `--biased`: Resamples the dataset so that 90% of the data is signal and 10% is background.
- `--save_every_n_files`: Number of ntuples to save per preprocessed file.

## Training the Model
To run a training session, use the `train_ECON_AE_CAE.py` script. Below is an example command:

```bash
python train_ECON_AE_CAE.py --opath test_new_run --mname test --model_per_eLink --alloc_geom old --data_path test_data_saving --loss tele --optim lion --lr 1e-4 --lr_sched cos --train_dataset_size 2000 --test_dataset_size 1000 --val_dataset_size 1000 --batchsize 128 --num_files 1 --nepochs 10
```

- `--opath`: Output directory for the training run.
- `--mname`: Model name.
- `--model_per_eLink`: Trains a unique CAE per possible eLink allocation.
- `--alloc_geom`: Allocation geometry.
- `--data_path`: Path to the preprocessed data.
- `--loss`: Loss function to use.
- `--optim`: Optimizer to use.
- `--lr`: Learning rate.
- `--lr_sched`: Learning rate scheduler.
- `--train_dataset_size`: Size of the training dataset.
- `--test_dataset_size`: Size of the test dataset.
- `--val_dataset_size`: Size of the validation dataset.
- `--batchsize`: Batch size.
- `--num_files`: Number of files to use.
- `--nepochs`: Number of epochs.

## File Descriptions
- `gen_latent_samples.py`: Script to generate latent samples.
- `process_data.py`: Script to process raw data and generate datasets.
- `train_ECON_AE_CAE.py`: Script to train the ECON Autoencoder models.
- `preprocess_CMSSW.py`: Script to preprocess CMSSW models.
- `utils/graph.py`: Utility functions for graph operations.
- `utils/utils.py`: General utility functions.
- `utils/telescope.py`: Functions related to telescope operations.
- `utils/files.py`: Functions for file operations.

## Additional Information
For more detailed information on each script and its usage, please refer to the inline comments and docstrings within the code files.
