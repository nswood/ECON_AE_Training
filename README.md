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
python process_data.py --opath test_data_saving --num_files 2 --model_per_eLink --biased 0.90 --save_every_n_files 1 --alloc_geom old
```

Arguments:
- `--opath`: Output directory for saved data.
- `--num_files`: Number of ntuples to preprocess.
- `--model_per_eLink`: Trains a unique CAE per possible eLink allocation.
- `--model_per_bit_config`: Trains a unique CAE per possible bit allocation.
- `--biased`: Resamples the dataset so that n% of the data is signal and (1-n)% is background (specify n as a float).
- `--save_every_n_files`: Number of ntuples to combine per preprocessed output file.
- `--alloc_geom`: The allocation geometry (old, new).

## Training the Model
Use the `train_ECON_AE_CAE.py` script to train the model. The `train_ECON_AE_CAE.py` script automatically runs `preprocess_CMSSW.py` which generates the necessary files to run the trained CAE in CMSSW. Below is an example command:

```bash
python train_ECON_AE_CAE.py --opath test_new_run --mname test --model_per_eLink --alloc_geom old --data_path test_data_saving --loss tele --optim lion --lr 1e-4 --lr_sched cos --train_dataset_size 2000 --test_dataset_size 1000 --val_dataset_size 1000 --batchsize 128 --num_files 1 --nepochs 10
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


## Hyperparameter Scan

The hyperparameter scan is implemented in `run_hyperband_search.py` and uses Keras‑Tuner’s Hyperband strategy to:

1. **Search** for the best hyperparameters (learning rate, optimizer, weight decay, batch size, LR scheduler, etc.).
2. **Refine** the best configuration by retraining over multiple random seeds.
3. **Finalize** training on a larger dataset.
4. **Export** both standard Keras models and CMSSW‑compatible versions.

---

### Quick Test Run

Use `--test_run` to exercise the full pipeline in ~5 minutes:

```bash
python run_hyperband_search.py \
  --opath test_hyperband_search \
  --mname search \
  --model_per_eLink \
  --specific_m 5 \
  --data_path /path/to/data \
  --test_run
```

This will create:

```
test_hyperband_search/
└── hyperband_search_5/
    ├── cae_hyperband_base_dir/                    # TensorBoard root for all trials
    ├── log/                                       # Per-trial & final logs
    │   ├── trial_0/
    │   ├── trial_1/
    │   └── final/
    │       ├── best_hyperparameters.csv           # HPs + best_val_loss + best_seed
    │       ├── performance_records.csv            # val_loss vs seed
    │       └── final_val_loss.csv                 # seed vs larger‑dataset loss
    ├── best_model_eLink_5_post_seed_variation/    # Best model + weights
    ├── best_model_eLink_5_post_seed_variation_for_CMSSW/
    ├── best_model_eLink_5_post_seed_variation_larger_dataset/
    └── best_model_eLink_5_post_seed_variation_larger_dataset_for_CMSSW/
```

---

### Full Usage

```bash
python run_hyperband_search.py \
  --opath <output_path>            # root dir for all outputs
  --mname <model_name>             # prefix when exporting models
  --model_per_eLink                 # scan mode: one model per eLink index
  --specific_m <eLink_index>       # which eLink to scan (2, 3, 4, or 5)
  --data_path <path/to/data>       # where your preprocessed datasets live
  [--test_run]                     # quick 2‑epoch smoke test
  [--skip_to_final_model]          # skip HP search, go straight to final training
  [--just_write_best_hyperparameters]  # only dump best HPs to CSV
  [--num_trials <int>]             # how many Hyperband trials (default 50)
  [--num_seeds <int>]              # how many seeds in seed‑search (default 20)
  [--orthogonal_regularization_factor <float>]  # orthogonal regularization factor (<0: regularization factor tunable, 0: no orthogonal regularization, >0: fixed regularization factor)
```

---

### What Happens Under the Hood

1. **Hyperband Search**  
   - **Hyperparameters**  
     - `lr` (log‑uniform between 1e‑5 and 1e‑2)  
     - `optimizer` (`adam` or `lion`)  
     - `weight_decay` (1e‑6 to 1e‑2)  
     - `batch_size` (64, 128, 256, 512, 1024)  
     - `lr_scheduler` (cosine, warm restarts, step decay, exponential decay)  
     - *(optional)* `orthogonal_regularization_factor`  
   - Logs each trial to `cae_hyperband_base_dir` inside your `--opath`.

2. **Seed Variation**  
   - Retrains the best HP configuration over `--num_seeds` random seeds to find the single seed that minimizes validation loss.  
   - Saves per‑seed results to `log/final/performance_records.csv`.

3. **Larger‑Dataset Training**  
   - Takes the best HP + seed, rebuilds the model, and trains for up to 100 epochs on a much larger hold‑out dataset (500 K samples; 25 K if `--test_run`).  
   - Records the final validation loss in `final_val_loss.csv`.

4. **Model Export**  
   - **Keras**: Saved under  
     `best_model_eLink_<m>_post_seed_variation[_larger_dataset]`  
   - **CMSSW**: Converted via  
     `utils.fix_preprocess_CMSSW.save_CMSSW_compatible_model(...)`  
     and saved in parallel folders ending in `_for_CMSSW`.


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
