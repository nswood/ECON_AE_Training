import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.layers import Layer
from qkeras import QActivation,QConv2D,QDense,quantized_bits
import qkeras
from qkeras.utils import model_save_quantized_weights
from keras.models import Model
from keras.layers import *
from telescope import *
from utils import *
import inspect
import json

import os
import sys
import graph

import pickle
from tensorflow.keras.models import model_from_json
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

import matplotlib.pyplot as plt
import mplhep as hep

def filter_for_flat_distribution(dataset, index_i):
    """
    Filters the given TensorFlow dataset to achieve a flat distribution over the specified index i
    of the second element (assumed to be an 8-dimensional tensor) in each dataset element.

    Args:
    - dataset (tf.data.Dataset): The input dataset.
    - index_i (int): The index of the 8-dimensional tensor to achieve a flat distribution over.

    Returns:
    - tf.data.Dataset: A new dataset filtered to achieve a flat distribution across non-zero bins for index_i.
    """
    # Extract the values at index_i from the dataset
    values_to_balance = np.array(list(dataset.map(lambda features, labels: labels[index_i]).as_numpy_iterator()))
    
    # Compute histogram over these values
    counts, bins = np.histogram(values_to_balance, bins=10)
    
    # Identify non-zero bins and determine the minimum count across them for a flat distribution
    non_zero_bins = counts > 0
    min_count_in_non_zero_bins = np.min(counts[non_zero_bins])
    
    # Determine which indices to include for a flat distribution
    indices_to_include = []
    current_counts = np.zeros_like(counts)
    for i, value in enumerate(values_to_balance):
        bin_index = np.digitize(value, bins) - 1
        bin_index = min(bin_index, len(current_counts) - 1)  # Ensure bin_index is within bounds
        if current_counts[bin_index] < min_count_in_non_zero_bins:
            indices_to_include.append(i)
            current_counts[bin_index] += 1
            
    # Convert list of indices to a TensorFlow constant for filtering
    indices_to_include_tf = tf.constant(indices_to_include, dtype=tf.int64)
    
    # Filtering function to apply with the dataset's enumerate method
    def filter_func(index, data):
        return tf.reduce_any(tf.equal(indices_to_include_tf, index))
    
    # Apply filtering to achieve the flat distribution
    filtered_dataset = dataset.enumerate().filter(filter_func).map(lambda idx, data: data)
    
    return filtered_dataset

p = ArgumentParser()
p.add_args(
    ('--mname', p.STR),
    ('--loss', p.STR), ('--nepochs', p.INT),
    ('--opath', p.STR),
    ('--mpath', p.STR),('--prepath', p.STR),('--continue_training', p.STORE_TRUE), ('--batchsize', p.INT),
    ('--lr', {'type': float}),
    ('--num_files', p.INT),('--optim', p.STR),('--model_per_eLink',  p.STORE_TRUE),('--model_per_bit_config',  p.STORE_TRUE),('--biased', {'type': float}), ('--alloc_geom', p.STR),('--low_eta_ft', p.STORE_TRUE),('--all_pileup', p.STORE_TRUE)
     
    
)

    
remap_8x8 = [ 4, 12, 20, 28,  5, 13, 21, 29,  6, 14, 22, 30,  7, 15, 23, 31, 
              24, 25, 26, 27, 16, 17, 18, 19,  8,  9, 10, 11,  0,  1,  2,  3, 
              59, 51, 43, 35, 58, 50, 42, 34, 57, 49, 41, 33, 56, 48, 40, 32]

with open('eLink_filts.pkl', 'rb') as f:
    key_df = pickle.load(f)

    
 

    
def get_pams():
    jsonpams={}      
    opt_classes = tuple(opt[1] for opt in inspect.getmembers(tf.keras.optimizers,inspect.isclass))
    for k,v in self.pams.items():
        if type(v)==type(np.array([])):
            jsonpams[k] = v.tolist()
        elif  isinstance(v,opt_classes):
            config = {}
            for hp in v.get_config():
                config[hp] = str(v.get_config()[hp])
            jsonpams[k] = config
        elif  type(v)==type(telescopeMSE8x8):
            jsonpams[k] =str(v) 
        else:
            jsonpams[k] = v 
    return jsonpams

def save_models(autoencoder, name, isQK=False):

    json_string = autoencoder.to_json()
    encoder = autoencoder.get_layer("encoder")
    decoder = autoencoder.get_layer("decoder")
    f'./{model_dir}/{name}.json'
    with open(f'./{model_dir}/{name}.json','w') as f:        f.write(autoencoder.to_json())
    with open(f'./{model_dir}/encoder_{name}.json','w') as f:            f.write(encoder.to_json())
    with open(f'./{model_dir}/decoder_{name}.json','w') as f:            f.write(decoder.to_json())

    autoencoder.save_weights(f'./{model_dir}/{name}.hdf5')
    encoder.save_weights(f'./{model_dir}/encoder_{name}.hdf5')
    decoder.save_weights(f'./{model_dir}/decoder_{name}.hdf5')
    if isQK:
        encoder_qWeight = model_save_quantized_weights(encoder)
        with open(f'{model_dir}/encoder_{name}.pkl','wb') as f:
            pickle.dump(encoder_qWeight,f)
        encoder = graph.set_quantized_weights(encoder,f'{model_dir}/encoder_'+name+'.pkl')
    graph.write_frozen_graph_enc(encoder,'encoder_'+name+'.pb',logdir = model_dir)
    graph.write_frozen_graph_enc(encoder,'encoder_'+name+'.pb.ascii',logdir = model_dir,asText=True)
    graph.write_frozen_graph_dec(decoder,'decoder_'+name+'.pb',logdir = model_dir)
    graph.write_frozen_graph_dec(decoder,'decoder_'+name+'.pb.ascii',logdir = model_dir,asText=True)

    graph.plot_weights(autoencoder,outdir = model_dir)
    graph.plot_weights(encoder,outdir = model_dir)
    graph.plot_weights(decoder,outdir = model_dir)
    


def load_matching_state_dict(model, state_dict_path):
    state_dict = tf.compat.v1.train.load_checkpoint(state_dict_path)
    model_variables = model.trainable_variables
    filtered_state_dict = {}
    for var in model_variables:
        var_name = var.name.split(':')[0]
        if var_name in state_dict:
            filtered_state_dict[var_name] = state_dict[var_name]
    tf.compat.v1.train.init_from_checkpoint(state_dict_path, filtered_state_dict)
    
    
def mean_mse_loss(y_true, y_pred):
    
    max_values = tf.reduce_max(y_true[:,], axis=1)
    
    y_true = tf.gather(K.reshape(y_true,(-1,64)),remap_8x8,axis=-1)
    y_pred = tf.gather(K.reshape(y_pred,(-1,64)),remap_8x8,axis=-1)
    # Calculate the squared difference between predicted and target values
    squared_diff = tf.square(y_pred - y_true)

    # Calculate the MSE per row (reduce_mean along axis=1)
    mse_per_row = tf.reduce_mean(squared_diff, axis=1)
    weighted_mse_per_row = mse_per_row * max_values
    
    # Take the mean of the MSE values to get the overall MSE loss
    mean_mse_loss = tf.reduce_mean(weighted_mse_per_row)
    return mean_mse_loss

def resample_indices(indices, energy, bin_edges, target_count, bin_index):
    bin_indices = indices[(energy > bin_edges[bin_index]) & (energy <= bin_edges[bin_index+1])]
    if len(bin_indices) > target_count:
        return np.random.choice(bin_indices, size=target_count, replace=False)
    else:
        return np.random.choice(bin_indices, size=target_count, replace=True)

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
def custom_resample(wafers,c,simE):
    
    label = (simE[:,0] != 0).astype(int)
    n = len(label)
    print(Counter(label))
    indices = np.expand_dims(np.arange(n),axis = -1)
    # 10x upsample signal
    over = RandomOverSampler(sampling_strategy=0.1)
    indices_p, label_p = over.fit_resample(indices, label)
    # downsample until 1:2::pilup:signal
    signal_percent = 1-args.biased
    ratio = args.biased / signal_percent
    if ratio > 1:
        ratio = 1 / ratio
        under = RandomUnderSampler(sampling_strategy=ratio)
        indices_p, label_p = under.fit_resample(indices_p, label_p)
    else:
        under = RandomUnderSampler(sampling_strategy=ratio)
        indices_p, label_p = under.fit_resample(indices_p, label_p)
    print(Counter(label_p))
    wafers_p = wafers[indices_p[:,0]]
    c_p = c[indices_p[:,0]]
    
    return wafers_p, c_p

def get_old_mask(eLinks, df):
    # Initialize a mask with all False values, with the same index as the DataFrame
    mask = pd.Series([False] * len(df), index=df.index)
    
    for eLink in eLinks:
        if eLink == 5:
            mask = mask | ((df['layer'] <= 11) & (df['layer'] >= 5))
        elif eLink == 4:
            mask = mask | ((df['layer'] == 7) | (df['layer'] == 11))
        elif eLink == 3:
            mask = mask | (df['layer'] == 13)
        elif eLink == 2:
            mask = mask | ((df['layer'] < 7) | (df['layer'] > 13))
        elif eLink == -1:
            mask = mask | (df['layer'] > 0)
    
    return mask

    
def load_data(nfiles,batchsize,model_info = -1, normalize = True):
    from files import get_rootfiles
    from coffea.nanoevents import NanoEventsFactory
    import awkward as ak
    import numpy as np
    ecr = np.vectorize(encode)
    data_list = []
    simE_list = []
    if args.model_per_eLink:
        eLinks = model_info
    elif args.model_per_bit_config:
        bitsPerOutput = model_info

    # Paths to Simon's dataset
    hostid = 'cmseos.fnal.gov'
    basepath = '/store/group/lpcpfnano/srothman/Nov08_2023_ECON_trainingdata'
    tree = 'FloatingpointThreshold0DummyHistomaxDummynTuple/HGCalTriggerNtuple'

    files = get_rootfiles(hostid, basepath)[0:nfiles]

    os.environ["XRD_REQUESTTIMEOUT"] = "120"
    os.environ["XRD_REDIRECTTIMEOUT"] = "120"
    #loop over all the files
    for i,file in enumerate(files):
        x = NanoEventsFactory.from_root(file, treepath=tree).events()

        min_pt = -1  # replace with your minimum value
        max_pt = 10e10  # replace with your maximum value
        gen_pt = ak.to_pandas(x.gen.pt).groupby(level=0).mean()
        mask = (gen_pt['values'] >= min_pt) & (gen_pt['values'] <= max_pt)
        
        
        layer = ak.to_pandas(x.wafer.layer)
        eta = ak.to_pandas(x.wafer.eta)
        v = ak.to_pandas(x.wafer.waferv)
        u = ak.to_pandas(x.wafer.waferu)
        wafertype = ak.to_pandas(x.wafer.wafertype)
        wafer_sim_energy = ak.to_pandas(x.wafer.simenergy)
        wafer_energy = ak.to_pandas(x.wafer.energy)
        
        # Combine all DataFrames into a single DataFrame
        data_dict = {
            'eta': eta.values.flatten(),
            'v': v.values.flatten(),
            'u': u.values.flatten(),
            'wafertype': wafertype.values.flatten(),
            'wafer_sim_energy': wafer_sim_energy.values.flatten(),
            'wafer_energy': wafer_energy.values.flatten(),
            'layer': layer.values.flatten()
        }

        # Add additional features AEin1 to AEin63 to the data dictionary
        key = 'AEin0'
        data_dict[key] = ak.to_pandas(x.wafer[key]).values.flatten()
        for i in range(1, 64):
            key = f'AEin{i}'
            data_dict[key] = ak.to_pandas(x.wafer[key]).values.flatten()
            key = f'CALQ{int(i)}'
            data_dict[key] = ak.to_pandas(x.wafer[key]).values.flatten()
            
        
        # Combine all data into a single DataFrame
        combined_df = pd.DataFrame(data_dict, index=eta.index)
        calq_columns = [f'CALQ{i}' for i in range(1,64)]
        combined_df['sumCALQ'] = combined_df[calq_columns].sum(axis=1)
        
        if args.alloc_geom == 'new':
            if args.model_per_eLink:
                filtered_key_df = key_df[key_df['trigLinks'] == float(eLinks)]
                filtered_df = pd.merge(combined_df, filtered_key_df[['u', 'v', 'layer']], on=['u', 'v', 'layer'], how='inner')
            elif args.model_per_bit_config:
                eLinks_with_bit_alloc = [index for index, value in enumerate(bitsPerOutputLink) if value == bitsPerOutput]
                eLinks_with_bit_alloc = [float(b) for b in eLinks_with_bit_alloc if b < 12]
                filtered_key_df = key_df[key_df['trigLinks'].isin(eLinks_with_bit_alloc)]
                filtered_df = pd.merge(combined_df, filtered_key_df[['u', 'v', 'layer']], on=['u', 'v', 'layer'], how='inner')
        
        elif args.alloc_geom =='old':
            if args.model_per_eLink:
                mask = get_old_mask(eLinks, combined_df)
                filtered_df = combined_df[mask]
            elif args.model_per_bit_config:
                eLinks_with_bit_alloc = [index for index, value in enumerate(bitsPerOutputLink) if value == bitsPerOutput]
                eLinks_with_bit_alloc = [b for b in eLinks_with_bit_alloc if b < 6]
                
                mask = get_old_mask(eLinks_with_bit_alloc, combined_df)
#                 print(combined_df)
                filtered_df = combined_df[mask]
                filtered_df = filtered_df.copy()
#                 print(filtered_df)
                
                
        print('Size after eLink filtering')
        print(len(filtered_df))
        
        # Process the filtered DataFrame
        filtered_df.loc[:,'eta'] = filtered_df['eta'] / 3.1
        filtered_df.loc[:,'v'] = filtered_df['v'] / 12
        filtered_df.loc[:,'u'] = filtered_df['u'] / 12
        filtered_df.loc[:,'layer'] = (filtered_df['layer']-1) / 46

        # Convert wafertype to one-hot encoding
        temp = filtered_df['wafertype'].astype(int).to_numpy()
        wafertype_one_hot = np.zeros((temp.size, 3))
        wafertype_one_hot[np.arange(temp.size), temp] = 1

        # Assign the processed columns back to the DataFrame
        filtered_df['wafertype'] = list(wafertype_one_hot)
        filtered_df['sumCALQ'] = np.squeeze(filtered_df['sumCALQ'].to_numpy())
        filtered_df['wafer_sim_energy'] = np.squeeze(filtered_df['wafer_sim_energy'].to_numpy())
        filtered_df['wafer_energy'] = np.squeeze(filtered_df['wafer_energy'].to_numpy())
        filtered_df['layer'] = np.squeeze(filtered_df['layer'].to_numpy())
        

        inputs = []
        for i in range(64):
            cur = filtered_df['AEin%d'%i]
            cur = np.squeeze(cur.to_numpy())
            inputs.append(cur) 
        inputs = np.stack(inputs, axis=-1) #stack all 64 inputs
        inputs = np.reshape(inputs, (-1, 8, 8))
        
        
        layer = filtered_df['layer'].to_numpy()
        eta = filtered_df['eta'].to_numpy()
        v = filtered_df['v'].to_numpy()
        u = filtered_df['u'].to_numpy()
        wafertype = np.array(filtered_df['wafertype'].tolist())
        sumCALQ = filtered_df['sumCALQ'].to_numpy()
        sumCALQ = np.log(sumCALQ+1)
        wafer_sim_energy = filtered_df['wafer_sim_energy'].to_numpy()
        wafer_energy = filtered_df['wafer_energy'].to_numpy()
        data_list.append([inputs,eta,v,u,wafertype,sumCALQ,layer])
        simE_list.append(wafer_sim_energy)
        
         

    
    inputs_list = []
    eta_list = []
    v_list = []
    u_list = []
    wafertype_list = []
    sumCALQ_list = []
    layer_list = []
    
    for item in data_list:
        inputs, eta, v, u, wafertype, sumCALQ,layer = item
        inputs_list.append(inputs)
        eta_list.append(eta)
        v_list.append(v)
        u_list.append(u)
        wafertype_list.append(wafertype)
        sumCALQ_list.append(sumCALQ)
        layer_list.append(layer)

    concatenated_inputs = np.expand_dims(np.concatenate(inputs_list),axis = -1)
    concatenated_eta = np.expand_dims(np.concatenate(eta_list),axis = -1)
    concatenated_v = np.expand_dims(np.concatenate(v_list),axis = -1)
    concatenated_u = np.expand_dims(np.concatenate(u_list),axis = -1)
    concatenated_wafertype = np.concatenate(wafertype_list)
    concatenated_sumCALQ = np.expand_dims(np.concatenate(sumCALQ_list),axis = -1)
    concatenated_layer = np.expand_dims(np.concatenate(layer_list),axis = -1)
    concatenated_simE = np.expand_dims(np.concatenate(simE_list),axis = -1)
    concatenated_cond = np.hstack([concatenated_eta,concatenated_v,concatenated_u, concatenated_wafertype, concatenated_sumCALQ,concatenated_layer])
    
    if args.low_eta_ft:
        print(f'Data before eta fine tuning:{len(concatenated_cond)}')
        mask = (concatenated_eta < 2.1/3.1)[:,0]
        print(mask)
        concatenated_cond = concatenated_cond[mask]
        concatenated_inputs = concatenated_inputs[mask]
        print(f'Data after eta fine tuning:{len(concatenated_cond)}')
        
    events = int(np.min([len(concatenated_cond), 10000000]))
    indices = np.random.permutation(events)
    # Calculate 80% of n
    num_selected = int(0.8 * events)

    # Select the first 80% of the indices
    train_indices = indices[:num_selected]
    test_indices = indices[num_selected:]
    wafer_train = concatenated_inputs[train_indices]
    wafer_test = concatenated_inputs[test_indices]
    
    simE_train = concatenated_simE[train_indices]
    simE_test = concatenated_simE[test_indices]

    cond_train = concatenated_cond[train_indices]
    cond_test = concatenated_cond[test_indices]
    if args.biased:
        wafer_train,cond_train = custom_resample(wafer_train,cond_train,simE_train)
        print(wafer_train.shape)
        wafer_test,cond_test = custom_resample(wafer_test,cond_test, simE_test)
    if args.all_pileup:
        mask_train = (simE_train[:,0] == 0)
        mask_test = (simE_test[:,0] == 0)
        print(len(cond_train))
        wafer_train,cond_train = wafer_train[mask_train],cond_train[mask_train]
        wafer_test,cond_test = wafer_test[mask_test],cond_test[mask_test]
        print(len(cond_train))
    
    # Create the training dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((wafer_train,cond_train)
    )

    # Create the test dataset
    test_dataset = tf.data.Dataset.from_tensor_slices((wafer_test,cond_test)
    )

    train_loader = train_dataset.batch(batchsize).shuffle(buffer_size=num_selected).prefetch(buffer_size=tf.data.AUTOTUNE)

    test_loader = test_dataset.batch(batchsize).shuffle(buffer_size=events-num_selected).prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_loader, test_loader
    
class keras_pad(Layer):
    def call(self, x):
        padding = tf.constant([[0,0],[0, 1], [0, 1], [0, 0]])
        return tf.pad(
        x, padding, mode='CONSTANT', constant_values=0, name=None
    )
    
    
class keras_minimum(Layer):
    def call(self, x, sat_val = 1):
        return tf.minimum(x,sat_val)
    
    
class keras_floor(Layer):
    def call(self, x):
        if isinstance(x, tf.SparseTensor):
            x = tf.sparse.to_dense(x)
            
        return tf.math.floor(x)
      
    
    
    
args = p.parse_args()
model_dir = args.opath


if not os.path.exists(model_dir):
    os.system("mkdir -p "+model_dir)

# Loop through each number of eLinks

if args.model_per_eLink:
    if args.alloc_geom == 'old':
        all_models = [2,3,4,5]
    elif args.alloc_geom =='new':
        all_models = [1,2,3,4,5,6,7,8,9,10,11]
elif args.model_per_bit_config:
    if args.alloc_geom == 'old':
        all_models = [3,5,7,9]
    elif args.alloc_geom =='new':
        all_models = [1,3,5,7,9]


bitsPerOutputLink = [0, 1, 3, 5, 7, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9]    

for m in all_models:
    if args.model_per_eLink:
        eLinks = m
        bitsPerOutput = bitsPerOutputLink[eLinks]
        print(f'Training Model with {eLinks} eLinks')
        model_dir = os.path.join(args.opath, f'model_{eLinks}_eLinks')
    elif args.model_per_bit_config:
        bitsPerOutput = m
        print(f'Training Model with {bitsPerOutput} output bits')
        model_dir = os.path.join(args.opath, f'model_{bitsPerOutput}_bits')
    
    if not os.path.exists(model_dir):
        os.system("mkdir -p " + model_dir)
    
    nIntegerBits = 1;
    nDecimalBits = bitsPerOutput - nIntegerBits;
    outputSaturationValue = (1 << nIntegerBits) - 1./(1 << nDecimalBits);
    maxBitsPerOutput = 9
    outputMaxIntSize = 1

    if bitsPerOutput > 0:
        outputMaxIntSize = 1 << nDecimalBits

    outputMaxIntSizeGlobal = 1
    if maxBitsPerOutput > 0:
        outputMaxIntSizeGlobal = 1 << (maxBitsPerOutput - nIntegerBits)

    batch = args.batchsize

    n_kernels = 8
    n_encoded=16
    conv_weightBits  = 6 
    conv_biasBits  = 6 
    dense_weightBits  = 6 
    dense_biasBits  = 6 
    encodedBits = 9
    CNN_kernel_size = 3
    


    input_enc = Input(batch_shape=(batch,8,8, 1), name = 'Wafer')
    # sum_input quantization is done in the dataloading step for simplicity
    
    cond = Input(batch_shape=(batch, 8), name = 'Cond')


    # Quantizing input, 8 bit quantization, 1 bit for integer
    x = QActivation(quantized_bits(bits = 8, integer = 1),name = 'input_quantization')(input_enc)
    x = keras_pad()(x)
    x = QConv2D(n_kernels,
                CNN_kernel_size, 
                strides=2,padding = 'valid', kernel_quantizer=quantized_bits(bits=conv_weightBits,integer=0,keep_negative=1,alpha=1), bias_quantizer=quantized_bits(bits=conv_biasBits,integer=0,keep_negative=1,alpha=1),
                name="conv2d")(x)
    
    x = QActivation(quantized_bits(bits = 8, integer = 1),name = 'act')(x)
    

    x = Flatten()(x)

    x = QDense(n_encoded, 
               kernel_quantizer=quantized_bits(bits=dense_weightBits,integer=0,keep_negative=1,alpha=1),
               bias_quantizer=quantized_bits(bits=dense_biasBits,integer=0,keep_negative=1,alpha=1),
               name="dense")(x)

    # Quantizing latent space, 9 bit quantization, 1 bit for integer
    x = QActivation(qkeras.quantized_bits(bits = 9, integer = 1),name = 'latent_quantization')(x)
    latent = x
    if bitsPerOutput > 0 and maxBitsPerOutput > 0:
        latent = keras_floor()(latent *  outputMaxIntSize)
        latent = keras_minimum()(latent/outputMaxIntSize, sat_val = outputSaturationValue)

    latent = concatenate([latent,cond],axis=1)
   

    
    input_dec = Input(batch_shape=(batch,24))
    y = Dense(24)(input_dec)
    y = ReLU()(y)
    y = Dense(64)(y)
    y = ReLU()(y)
    y = Dense(128)(y)
    y = ReLU()(y)
    y = Reshape((4, 4, 8))(y)
    y = Conv2DTranspose(1, (3, 3), strides=(2, 2),padding = 'valid')(y)
    y =y[:,0:8,0:8]
    y = ReLU()(y)
    recon = y
    
    encoder = keras.Model([input_enc,cond], latent, name="encoder")
    decoder = keras.Model([input_dec], recon, name="decoder")

    cae = Model(
        inputs=[input_enc,cond],
        outputs=decoder([encoder([input_enc,cond])]),
        name="cae"
    )

    if args.loss == 'mse':
        loss=mean_mse_loss
    elif args.loss == 'tele':
        print('Using tele')
        loss = telescopeMSE8x8
    elif args.loss == 'emd':
        loss = get_emd_loss(args.emd_pth)

    print(args.optim)
       
    if args.optim == 'adam':
        print('Using ADAM Optimizer')
        opt = tf.keras.optimizers.Adam(learning_rate = args.lr,weight_decay = 0.000025)
    elif args.optim == 'lion':
        print('Using Lion Optimizer')
        opt = tf.keras.optimizers.Lion(learning_rate = args.lr,weight_decay = 0.00025)

    cae.compile(optimizer=opt, loss=loss)
    cae.summary()
    def cosine_annealing(epoch, total_epochs, initial_lr):
        """Cosine annealing scheduler."""
        cos_inner = np.pi * (epoch % (total_epochs // 10))
        cos_inner /= total_epochs // 10
        cos_out = np.cos(cos_inner) + 1
        return float(initial_lr / 2 * cos_out)

    
    initial_lr = args.lr
    total_epochs = args.nepochs

    # Create a learning rate scheduler callback
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(
        lambda epoch: cosine_annealing(epoch, total_epochs, initial_lr)
    )

    # Loading Model
    if args.continue_training:
        
        cae.load_weights(os.path.join(model_dir, 'best-epoch.tf'))
        start_epoch = 1
        print(f"Continuing training from best model...")






    print('Loading Data')
    train_loader, test_loader = load_data(args.num_files,batch,model_info =m)
    print('Data Loaded')


    best_val_loss = 1e9

    all_train_loss = []
    all_val_loss = []


    if args.continue_training:
        loss_dict = {'train_loss': pd.read_csv(os.path.join(model_dir,'df.csv'))['train_loss'].tolist(), 
     'val_loss': pd.read_csv(os.path.join(model_dir,'df.csv'))['val_loss'].tolist()}
        start_epoch = 1
    else:
        start_epoch = 1
        loss_dict = {'train_loss': [], 'val_loss': []}




    for epoch in range(start_epoch, args.nepochs):
        total_loss_train = 0
        new_lr = cosine_annealing(epoch, total_epochs, initial_lr)
#         print(new_lr)
        tf.keras.backend.set_value(opt.learning_rate, new_lr)

        for wafers, cond in train_loader:

            loss = cae.train_on_batch([wafers,cond], wafers)
            total_loss_train = total_loss_train + loss
        

        total_loss_val = 0 
        for wafers, cond in test_loader:


            loss = cae.test_on_batch([wafers, cond], wafers)

            total_loss_val = total_loss_val+loss


        total_loss_train = total_loss_train /(len(train_loader))
        total_loss_val = total_loss_val /(len(test_loader))
        if epoch % 5 == 0:
            print('Epoch {:03d}, Loss: {:.8f}, ValLoss: {:.8f}'.format(
                epoch, total_loss_train,  total_loss_val))

        loss_dict['train_loss'].append(total_loss_train)
        loss_dict['val_loss'].append(total_loss_val)
        df = pd.DataFrame.from_dict(loss_dict)
        
        plt.figure(figsize=(10, 6))
        plt.plot(df['train_loss'], label='Training Loss')
        plt.plot(df['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        # Saving the plot in the same directory as the loss CSV
        plot_path = f"{model_dir}/loss_plot.png"
        plt.savefig(plot_path)
        df.to_csv(f"{model_dir}/df.csv", index=False)
        

        if total_loss_val < best_val_loss:
            if epoch % 25 == 0:
                print('New Best Model')
            best_val_loss = total_loss_val
            cae.save_weights(os.path.join(model_dir, 'best-epoch.tf'.format(epoch)))
            encoder.save_weights(os.path.join(model_dir, 'best-encoder-epoch.tf'.format(epoch)))
            decoder.save_weights(os.path.join(model_dir, 'best-decoder-epoch.tf'.format(epoch)))
    save_models(cae,args.mname,isQK = True)
    
    

import subprocess

if args.model_per_eLink:
    args = [
        'python', 'dev_preCMSSW.py',
        '--mname', 'vanilla_AE',
        '--mpath', args.opath,
        '--model_per_eLink',
        '--alloc_geom', args.alloc_geom
        ]
elif args.model_per_bit_config:
    args = [
        'python', 'dev_preCMSSW.py',
        '--mname', 'vanilla_AE',
        '--mpath', args.opath,
        '--model_per_bit_config',
        '--alloc_geom', args.alloc_geom
        ]

# Run the other script with the arguments
subprocess.run(args)
