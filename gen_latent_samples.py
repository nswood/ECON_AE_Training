import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from qkeras import QActivation, QConv2D, QDense, quantized_bits
import qkeras
from qkeras.utils import model_save_quantized_weights
from keras.models import Model
from keras.layers import *
from utils.telescope import *
from utils.utils import *
import inspect
import json
import os
import sys
import utils.graph as graph

import pickle
from tensorflow.keras.models import model_from_json
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

import matplotlib.pyplot as plt
import mplhep as hep


p = ArgumentParser()
p.add_args(
    ('--mpath', p.STR),
    ('--opath', p.STR),
    
)

'''
---------------------------------------------------------------------

mpath: path to directory with all model folders (for this case: ~/trained_model) 

opath: where you want latent space variables to saved to. Will create a dir for them there
---------------------------------------------------------------------
'''    


    
remap_8x8 = [ 4, 12, 20, 28,  5, 13, 21, 29,  6, 14, 22, 30,  7, 15, 23, 31, 
              24, 25, 26, 27, 16, 17, 18, 19,  8,  9, 10, 11,  0,  1,  2,  3, 
              59, 51, 43, 35, 58, 50, 42, 34, 57, 49, 41, 33, 56, 48, 40, 32]


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


def load_matching_state_dict(model, state_dict_path):
    state_dict = tf.compat.v1.train.load_checkpoint(state_dict_path)
    model_variables = model.trainable_variables
    filtered_state_dict = {}
    for var in model_variables:
        var_name = var.name.split(':')[0]
        if var_name in state_dict:
            filtered_state_dict[var_name] = state_dict[var_name]
    tf.compat.v1.train.init_from_checkpoint(state_dict_path, filtered_state_dict)

def load_data(nfiles,batchsize,eLinks = -1, normalize = True):
    from files import get_rootfiles
    from coffea.nanoevents import NanoEventsFactory
    import awkward as ak
    import numpy as np
    ecr = np.vectorize(encode)
    data_list = []

    
    '''
    ---------------------------------------------------------------------
    
    Here you should implement some kind of data loading
    
    The rest of the code expects .root files in the same format as on LPC 
    
    ---------------------------------------------------------------------
    '''    
    # Paths to Simon's dataset
#     hostid = 'cmseos.fnal.gov'
#     basepath = '/store/group/lpcpfnano/srothman/Nov08_2023_ECON_trainingdata'
#     tree = 'FloatingpointThreshold0DummyHistomaxDummynTuple/HGCalTriggerNtuple'

    files = get_rootfiles(hostid, basepath)[0:nfiles]


    #loop over all the files
    for i,file in enumerate(files):
        
        # Replace this with loading Simon's data from your storage
        x = NanoEventsFactory.from_root(file, treepath=tree).events()

        min_pt = 10  # replace with your minimum value
        max_pt = 1000  # replace with your maximum value
        gen_pt = ak.to_pandas(x.gen.pt).groupby(level=0).mean()
        mask = (gen_pt['values'] >= min_pt) & (gen_pt['values'] <= max_pt)
        layers = ak.to_pandas(x.wafer.layer)
        layers = layers.loc[layers.index.get_level_values('entry').isin(mask)]

        eta = ak.to_pandas(x.wafer.eta)
        eta = eta.loc[eta.index.get_level_values('entry').isin(mask)]

        waferv = ak.to_pandas(x.wafer.waferv)
        waferv = waferv.loc[waferv.index.get_level_values('entry').isin(mask)]

        waferu = ak.to_pandas(x.wafer.waferu)
        waferu = waferu.loc[waferu.index.get_level_values('entry').isin(mask)]

        wafertype = ak.to_pandas(x.wafer.wafertype)
        wafertype = wafertype.loc[wafertype.index.get_level_values('entry').isin(mask)]

        sumCALQ = ak.to_pandas(x.wafer['CALQ0'])
        sumCALQ = sumCALQ.loc[sumCALQ.index.get_level_values('entry').isin(mask)]


        wafer_sim_energy = ak.to_pandas(x.wafer.simenergy)
        wafer_sim_energy = wafer_sim_energy.loc[wafer_sim_energy.index.get_level_values('entry').isin(mask)]

        wafer_energy = ak.to_pandas(x.wafer.energy)
        wafer_energy = wafer_energy.loc[wafer_energy.index.get_level_values('entry').isin(mask)]


        # Nate's naive normalization scheme for inputs
        layers = np.squeeze(layers.to_numpy())
        eta = np.squeeze(eta.to_numpy())/3.1
        waferv = np.squeeze(waferv.to_numpy())/12
        waferu = np.squeeze(waferu.to_numpy())/12
        temp = np.squeeze(wafertype.to_numpy())
        wafertype = np.zeros((temp.size, temp.max() + 1))
        wafertype[np.arange(temp.size), temp] = 1
        sumCALQ = np.squeeze(sumCALQ.to_numpy())
        wafer_sim_energy = np.squeeze(wafer_sim_energy.to_numpy())
        wafer_energy = np.squeeze(wafer_energy.to_numpy())



        for i in range(1,64):
            cur = ak.to_pandas(x.wafer[f'CALQ{int(i)}'])
            cur = cur.loc[cur.index.get_level_values('entry').isin(mask)]
            cur = np.squeeze(cur.to_numpy())
            sumCALQ = sumCALQ + cur

        sumCALQ = np.log(sumCALQ+1)

        inputs = []
        for i in range(64):
            cur = ak.to_pandas(x.wafer['AEin%d'%i])
            cur = cur.loc[cur.index.get_level_values('entry').isin(mask)]
            cur = np.squeeze(cur.to_numpy())
            inputs.append(cur) 
        inputs = np.stack(inputs, axis=-1) #stack all 64 inputs
        inputs = np.reshape(inputs, (-1, 8, 8))
        select_eLinks = {5 : (layers<=11) & (layers>=5) ,
                 4 : (layers==7) | (layers==11),
                 3 : (layers==13),
                 2 : (layers<7) | (layers>13),
                 -1 : (layers>0)}
        inputs = inputs[select_eLinks[eLinks]]
        l =(layers[select_eLinks[eLinks]]-1)/(47-1)
        eta = eta[select_eLinks[eLinks]]
        waferv = waferv[select_eLinks[eLinks]]
        waferu = waferu[select_eLinks[eLinks]]
        wafertype = wafertype[select_eLinks[eLinks]]
        sumCALQ = sumCALQ[select_eLinks[eLinks]]
        wafer_sim_energy = wafer_sim_energy[select_eLinks[eLinks]]
        wafer_energy = wafer_energy[select_eLinks[eLinks]]
        
        # Could add some kind of preprocessing here if desired
        
        data_list.append([inputs,eta,waferv,waferu,wafertype,sumCALQ,l])


    inputs_list = []
    eta_list = []
    waferv_list = []
    waferu_list = []
    wafertype_list = []
    sumCALQ_list = []
    layer_list = []

    for item in data_list:
        inputs, eta, waferv, waferu, wafertype, sumCALQ,layers = item
        inputs_list.append(inputs)
        eta_list.append(eta)
        waferv_list.append(waferv)
        waferu_list.append(waferu)
        wafertype_list.append(wafertype)
        sumCALQ_list.append(sumCALQ)
        layer_list.append(layers)

    concatenated_inputs = np.expand_dims(np.concatenate(inputs_list),axis = -1)
    concatenated_eta = np.expand_dims(np.concatenate(eta_list),axis = -1)
    concatenated_waferv = np.expand_dims(np.concatenate(waferv_list),axis = -1)
    concatenated_waferu = np.expand_dims(np.concatenate(waferu_list),axis = -1)
    concatenated_wafertype = np.concatenate(wafertype_list)
    concatenated_sumCALQ = np.expand_dims(np.concatenate(sumCALQ_list),axis = -1)
    concatenated_layers = np.expand_dims(np.concatenate(layer_list),axis = -1)

    concatenated_cond = np.hstack([concatenated_eta,concatenated_waferv,concatenated_waferu, concatenated_wafertype, concatenated_sumCALQ,concatenated_layers])


    all_dataset = tf.data.Dataset.from_tensor_slices((concatenated_inputs, concatenated_cond)
    ).take(1000000)

    total_size = len(all_dataset)  # Replace with your dataset's total size
    print('total size: ',total_size)
    
    # Define your splitting ratio
    train_size = int(0.8 * total_size)
    test_size = total_size - train_size

    # Create the training dataset
    train_dataset = all_dataset.take(train_size)

    # Create the test dataset
    test_dataset = all_dataset.skip(train_size).take(test_size)

    train_loader = train_dataset.batch(batchsize).shuffle(buffer_size=train_size).prefetch(buffer_size=tf.data.AUTOTUNE)

    test_loader = test_dataset.batch(batchsize).shuffle(buffer_size=test_size).prefetch(buffer_size=tf.data.AUTOTUNE)


    return train_loader, test_loader    
    
    
args = p.parse_args()
model_dir = args.opath

if not os.path.exists(model_dir):
    os.system("mkdir -p "+model_dir)

for eLinks in [2,3,4,5]:
     
    bitsPerOutputLink = [0, 1, 3, 5, 7, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9]
    
    print(f'Processing Model with {eLinks} eLinks')
    model_dir = os.path.join(args.mpath, f'model_{eLinks}_eLinks')
    if not os.path.exists(model_dir):
        os.system("mkdir -p " + model_dir)
    bitsPerOutput = bitsPerOutputLink[eLinks]
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

    batch = 1

    n_kernels = 8
    n_encoded=16
    conv_weightBits  = 6 
    conv_biasBits  = 6 
    dense_weightBits  = 6 
    dense_biasBits  = 6 
    encodedBits = 9
    CNN_kernel_size = 3
    padding = tf.constant([[0,0],[0, 1], [0, 1], [0, 0]])


    input_enc = Input(batch_shape=(batch,8,8,1), name = 'Wafer')

    # Quantizing input, 8 bit quantization, 1 bit for integer
    x = QActivation(quantized_bits(bits = 8, integer = 1),name = 'input_quantization')(input_enc)
    x = tf.pad(
        x, padding, mode='CONSTANT', constant_values=0, name=None
    )
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
        latent = tf.minimum(tf.math.floor(latent *  outputMaxIntSize) /  outputMaxIntSize, outputSaturationValue)

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
    
    encoder = keras.Model([input_enc], latent, name="encoder")

    encoder_path = os.path.join(args.mpath,f'model_{eLinks}_eLinks','best-encoder-epoch.tf')
    encoder.load_weights(encoder_path)
    
    
    '''
    ---------------------------------------------------------------------
    
    Decoder weights are included, but not necessary for generating latent space variables so commented out
    
    ---------------------------------------------------------------------
    '''    
#     decoder = keras.Model([input_dec], recon, name="decoder")
#     decoder_path = os.path.join(args.mpath,f'model_{eLinks}_eLinks','best-decoder-epoch.tf')
#     decoder.load_weights(decoder_path)
    
    train_latent = []
    for wafers, cond in train_loader:
        train_latent.append(encoder.predict(wafers,cond))

    test_latent = []
    for wafers, cond in test_loader:
        test_latent.append(encoder.predict(wafers,cond))

    
    
    '''
    ---------------------------------------------------------------------
    
    Add some kind of saving of latent info in whatever format you want
    
    ---------------------------------------------------------------------
    '''
    
