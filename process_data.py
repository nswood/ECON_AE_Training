import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from coffea.nanoevents import NanoEventsFactory
import awkward as ak
from utils.files import get_rootfiles
from utils.utils import ArgumentParser, encode

import warnings

#Setting seed
SEED = 12345
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)



# Suppress *all* RuntimeWarnings from coffea.nanoevents.schemas.nanoaod
warnings.filterwarnings(
    action="ignore",
    category=RuntimeWarning,
    module=r"coffea\.nanoevents\.schemas\.nanoaod"
)

##############################################################################
# Argument Parser
##############################################################################
p = ArgumentParser()
p.add_args(
    ('--opath', p.STR),
    ('--num_files', p.INT),
    ('--model_per_eLink', p.STORE_TRUE),
    ('--model_per_bit_config', p.STORE_TRUE),
    ('--biased', {'type': float}),
    ('--save_every_n_files', p.INT),
    ('--alloc_geom', p.STR)
)
args = p.parse_args()

##############################################################################
# Global Configuration
##############################################################################
bitsPerOutputLink = [
    0, 1, 3, 5, 7, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9
]

# Make sure output directory exists
if not os.path.exists(args.opath):
    os.system("mkdir -p " + args.opath)

##############################################################################
# Resampling Function
##############################################################################
def custom_resample(wafers, c, simE):
    """
    Upsamples signal (simE != 0) by 10x, then undersamples
    to achieve a ratio of pileup : signal ~ (args.biased) : (1 - args.biased).
    """
    label = (simE[:, 0] != 0).astype(int)  # 1 for signal, 0 for pileup
    n = len(label)

    print("Original label distribution:", Counter(label))

    indices = np.expand_dims(np.arange(n), axis=-1)
    # 10x upsample signal
    over = RandomOverSampler(sampling_strategy=0.1)
    indices_p, label_p = over.fit_resample(indices, label)

    # Downsample until ratio = pileup : signal = args.biased : (1 - args.biased)
    signal_percent = 1 - args.biased
    ratio = args.biased / signal_percent

    if ratio > 1:
        ratio = 1 / ratio
        under = RandomUnderSampler(sampling_strategy=ratio)
        indices_p, label_p = under.fit_resample(indices_p, label_p)
    else:
        under = RandomUnderSampler(sampling_strategy=ratio)
        indices_p, label_p = under.fit_resample(indices_p, label_p)

    print("New label distribution:", Counter(label_p))

    # Apply final indices
    wafers_p = wafers[indices_p[:, 0]]
    c_p = c[indices_p[:, 0]]
    return wafers_p, c_p

##############################################################################
# Masking Function (Old Geometry)
##############################################################################
def get_old_mask(eLink, df):
    """
    Creates a boolean mask for the input DataFrame 'df' based on 'eLink'
    selection logic for the 'layer' column.
    """
    mask = pd.Series([False] * len(df), index=df.index)

    if eLink == 5:
        # layers 5 through 11
        mask |= ((df['layer'] <= 11) & (df['layer'] >= 5))
    elif eLink == 4:
        mask |= ((df['layer'] == 7) | (df['layer'] == 11))
    elif eLink == 3:
        mask |= (df['layer'] == 13)
    elif eLink == 2:
        mask |= ((df['layer'] < 7) | (df['layer'] > 13))
    elif eLink == -1:
        # All layers
        mask |= (df['layer'] > 0)

    return mask

##############################################################################
# Main Data-Saving Function
##############################################################################
def process_data(files, save_every_n_files, model_info=-1, normalize=True, model_dir=None):
    """
    Reads up to 'nfiles' root files, processes them,
    filters wafers by geometry, splits into train/test, 
    optionally re-samples, and saves each chunk to TF datasets.
    """
    os.environ["XRD_REQUESTTIMEOUT"] = "120"
    os.environ["XRD_REDIRECTTIMEOUT"] = "120"

    # Decide eLinks or bits logic
    if args.model_per_eLink:
        eLinks = model_info
    elif args.model_per_bit_config:
        bitsPerOutput = model_info
    tree = 'FloatingpointThreshold0DummyHistomaxDummynTuple/HGCalTriggerNtuple'


    data_list = []
    simE_list = []
    num_saved_files = 0

    for i, file in enumerate(files):
        print("======== FILE INDEX =", i, "========")

        # Read events using coffea
        events = NanoEventsFactory.from_root(file, treepath=tree).events()

        # Basic gen_pt filtering
        min_pt, max_pt = -1, 1e15
        gen_pt = ak.to_pandas(events.gen.pt).groupby(level=0).mean()
        mask = (gen_pt['values'] >= min_pt) & (gen_pt['values'] <= max_pt)

        # Extract relevant columns
        layer_pd = ak.to_pandas(events.wafer.layer)
        eta_pd = ak.to_pandas(events.wafer.eta)
        v_pd = ak.to_pandas(events.wafer.waferv)
        u_pd = ak.to_pandas(events.wafer.waferu)
        wafertype_pd = ak.to_pandas(events.wafer.wafertype)
        wafer_sim_energy_pd = ak.to_pandas(events.wafer.simenergy)
        wafer_energy_pd = ak.to_pandas(events.wafer.energy)

        # Build a combined DataFrame
        data_dict = {
            'eta': eta_pd.values.flatten(),
            'v': v_pd.values.flatten(),
            'u': u_pd.values.flatten(),
            'wafertype': wafertype_pd.values.flatten(),
            'wafer_sim_energy': wafer_sim_energy_pd.values.flatten(),
            'wafer_energy': wafer_energy_pd.values.flatten(),
            'layer': layer_pd.values.flatten()
        }

        # Add AEin0 as well as CALQ1..63, AEin1..63
        key = 'AEin0'
        data_dict[key] = ak.to_pandas(events.wafer[key]).values.flatten()
        for j in range(1, 64):
            # AEin
            key_ae = f'AEin{j}'
            data_dict[key_ae] = ak.to_pandas(events.wafer[key_ae]).values.flatten()

            # CALQ
            key_cq = f'CALQ{j}'
            data_dict[key_cq] = ak.to_pandas(events.wafer[key_cq]).values.flatten()

        combined_df = pd.DataFrame(data_dict, index=eta_pd.index)
        print("Size before eLink or geometry filtering:", len(combined_df))
        # Summation of CALQs
        calq_cols = [f'CALQ{i}' for i in range(1, 64)]
        combined_df['sumCALQ'] = combined_df[calq_cols].sum(axis=1)

        # Filter by geometry (alloc_geom)
        if args.alloc_geom == 'new':
            # You reference 'key_df' in the original code, but it isn't defined in this snippet.
            # If you have 'key_df' from a pickled file, load it above or adapt as necessary.
            if args.model_per_eLink:
                # Filter by eLinks in key_df
                # Example logic if you had key_df loaded somewhere else:
                # filtered_key_df = key_df[key_df['trigLinks'] == float(eLinks)]
                # combined_df = pd.merge(combined_df, filtered_key_df[['u', 'v', 'layer']], ...)
                pass
            elif args.model_per_bit_config:
                # Filter by bits
                pass
            filtered_df = combined_df
        elif args.alloc_geom == 'old':
            if args.model_per_eLink:
                # Use get_old_mask for the chosen eLinks
                mask_geom = get_old_mask(eLinks, combined_df)
                filtered_df = combined_df[mask_geom]
            elif args.model_per_bit_config:
                # Example: find eLinks that match bits, then get_old_mask
                pass
                filtered_df = combined_df
        else:
            filtered_df = combined_df

        print("Size after eLink or geometry filtering:", len(filtered_df))

        # Ensure filtered_df is a copy of the original DataFrame
        filtered_df = filtered_df.copy()

        # Modify the DataFrame in place using .loc
        filtered_df.loc[:, 'eta'] = filtered_df['eta'] / 3.1
        filtered_df.loc[:, 'v'] = filtered_df['v'] / 12
        filtered_df.loc[:, 'u'] = filtered_df['u'] / 12

        # One-hot encode wafertype
        temp = filtered_df['wafertype'].astype(int).to_numpy()
        wafertype_one_hot = np.zeros((temp.size, 3))
        wafertype_one_hot[np.arange(temp.size), temp] = 1
        filtered_df['wafertype'] = list(wafertype_one_hot)

        # Log transform sumCALQ
        filtered_df['sumCALQ'] = np.log(filtered_df['sumCALQ'] + 1)

        # Gather AEin(0..63) into 8x8
        inputs = []
        for j in range(64):
            col_name = f'AEin{j}'
            inputs.append(filtered_df[col_name].to_numpy())
        inputs = np.stack(inputs, axis=-1)  # shape: (N, 64)
        inputs = inputs.reshape(-1, 8, 8)

        # Build final arrays
        eta_arr = filtered_df['eta'].to_numpy()
        v_arr = filtered_df['v'].to_numpy()
        u_arr = filtered_df['u'].to_numpy()
        wafertype_arr = np.array(list(filtered_df['wafertype']))
        sumCALQ_arr = filtered_df['sumCALQ'].to_numpy()
        layer_arr = filtered_df['layer'].to_numpy()
        wafer_sim_energy_arr = filtered_df['wafer_sim_energy'].to_numpy()
        wafer_energy_arr = filtered_df['wafer_energy'].to_numpy()  # not used further, but kept

        data_list.append([inputs, eta_arr, v_arr, u_arr, wafertype_arr, sumCALQ_arr, layer_arr])
        simE_list.append(wafer_sim_energy_arr)

        # Every save_every_n_files files or at the last file, save a chunk
        if ((i % save_every_n_files == 0 and i != 0) or (i == len(files) - 1)):
            inputs_list = []
            eta_list = []
            v_list = []
            u_list = []
            wafertype_list = []
            sumCALQ_list = []
            layer_list = []

            # Consolidate
            for item in data_list:
                inputs_i, eta_i, v_i, u_i, wafertype_i, sumCALQ_i, layer_i = item
                inputs_list.append(inputs_i)
                eta_list.append(eta_i)
                v_list.append(v_i)
                u_list.append(u_i)
                wafertype_list.append(wafertype_i)
                sumCALQ_list.append(sumCALQ_i)
                layer_list.append(layer_i)

            # Expand dimension for inputs: final shape (N, 8, 8, 1)
            concatenated_inputs = np.expand_dims(np.concatenate(inputs_list), axis=-1)
            concatenated_eta = np.expand_dims(np.concatenate(eta_list), axis=-1)
            concatenated_v = np.expand_dims(np.concatenate(v_list), axis=-1)
            concatenated_u = np.expand_dims(np.concatenate(u_list), axis=-1)
            concatenated_wafertype = np.concatenate(wafertype_list)  # already multi-dim
            concatenated_sumCALQ = np.expand_dims(np.concatenate(sumCALQ_list), axis=-1)
            concatenated_layer = np.expand_dims(np.concatenate(layer_list), axis=-1)
            concatenated_simE = np.expand_dims(np.concatenate(simE_list), axis=-1)

            # Build condition array
            concatenated_cond = np.hstack([
                concatenated_eta,
                concatenated_v,
                concatenated_u,
                concatenated_wafertype,
                concatenated_sumCALQ,
                concatenated_layer
            ])

            # Shuffle, train/test split (80/20)
            events = len(concatenated_cond)
            indices = np.random.permutation(events)
            num_selected = int(0.8 * events)
            train_indices = indices[:num_selected]
            test_indices = indices[num_selected:]

            wafer_train = concatenated_inputs[train_indices]
            wafer_test = concatenated_inputs[test_indices]

            simE_train = concatenated_simE[train_indices]
            simE_test = concatenated_simE[test_indices]

            cond_train = concatenated_cond[train_indices]
            cond_test = concatenated_cond[test_indices]

            # Optionally re-sample
            if args.biased:
                wafer_train, cond_train = custom_resample(wafer_train, cond_train, simE_train)
                wafer_test, cond_test = custom_resample(wafer_test, cond_test, simE_test)

            # Build TF datasets
            train_dataset = tf.data.Dataset.from_tensor_slices((wafer_train, cond_train))
            test_dataset = tf.data.Dataset.from_tensor_slices((wafer_test, cond_test))

            # Save each chunk to disk
            save_path_train = os.path.join(model_dir, f"{num_saved_files}_train")
            save_path_test = os.path.join(model_dir, f"{num_saved_files}_test")
            tf.data.experimental.save(train_dataset, save_path_train)
            tf.data.experimental.save(test_dataset, save_path_test)

            # Reset accumulators
            data_list = []
            simE_list = []
            num_saved_files += 1


##############################################################################
# Main Loop Over eLinks or Bits
##############################################################################
if args.model_per_eLink:
    if args.alloc_geom == 'old':
        all_models = [2, 3, 4, 5]
    elif args.alloc_geom == 'new':
        all_models = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
elif args.model_per_bit_config:
    if args.alloc_geom == 'old':
        all_models = [3,5,7,9]  # Could be  if desired
    elif args.alloc_geom == 'new':
        all_models = [1, 3, 5, 7, 9]
else:
    all_models = []

for m in all_models:
    if args.model_per_eLink:
        eLinks = m
        print(f"=============================\n")
        print(f"Processing data with {m} eLinks\n")
        print(f"=============================\n")
        bitsPerOutput = bitsPerOutputLink[eLinks]
        model_dir = os.path.join(args.opath, f"data_{eLinks}_eLinks")
    elif args.model_per_bit_config:
        bitsPerOutput = m
        
        print(f"=============================\n")
        print(f"Processing data with {m} bits\n")
        print(f"=============================\n")
        model_dir = os.path.join(args.opath, f"data_{bitsPerOutput}_bits")
    else:
        model_dir = args.opath  # fallback

    # Ensure directory exists
    if not os.path.exists(model_dir):
        os.system("mkdir -p " + model_dir)

    all_files = get_rootfiles('cmsdata.phys.cmu.edu', '/store/user/eertorer/ECONAE/data/')
    all_files = sorted(all_files)
    process_data(all_files[:args.num_files], args.save_every_n_files, model_info=m, model_dir=model_dir)
