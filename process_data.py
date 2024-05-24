import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
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


p = ArgumentParser()
p.add_args(
    
    ('--opath', p.STR),('--b_percent', {'type': float}))
    

def load_data(normalize = True,eLinks = -1):
    from files import get_rootfiles
    from coffea.nanoevents import NanoEventsFactory
    import awkward as ak
    import numpy as np
    ecr = np.vectorize(encode)
    data_list = []
    

    hostid = 'cmseos.fnal.gov'
    basepath = '/store/group/lpcpfnano/srothman/Nov08_2023_ECON_trainingdata'
    tree = 'FloatingpointThreshold0DummyHistomaxDummynTuple/HGCalTriggerNtuple'
    
    files = get_rootfiles(hostid, basepath)[0:30]
    

    #loop over all the files
    for i,file in enumerate(files):
        x = NanoEventsFactory.from_root(file, treepath=tree).events()
        
        min_pt = 0 # replace with your minimum value
        max_pt = 100000  # replace with your maximum value
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

        
        
#         mask = (wafer_sim_energy > 0) 
#         indices_passing = np.where(mask)[0]
#         indices_not_passing = np.where(~mask)[0]
        
#         if args.b_percent is not None:
#             k = args.b_percent /(1-args.b_percent)
#         else: 
#             k = 3
#         desired_not_passing_count = int(len(indices_passing) / k) 
        
#         selected_not_passing_indices = np.random.choice(indices_not_passing, size=desired_not_passing_count, replace=False)

#         new_mask_indices = np.concatenate((indices_passing, selected_not_passing_indices))
#         mask = np.zeros_like(wafer_sim_energy, dtype=bool)
#         mask[new_mask_indices] = True
        

#         inputs = inputs[mask]
#         l =l[mask]
#         eta = eta[mask]
#         waferv = waferv[mask]
#         waferu = waferu[mask]
#         wafertype = wafertype[mask]
#         sumCALQ = sumCALQ[mask]
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

    # Final list of concatenated arrays
#     final_concatenated_list = [concatenated_inputs, concatenated_eta, concatenated_waferv, concatenated_waferu, concatenated_wafertype, concatenated_sumCALQ]
#     final_concatenated_list = [concatenated_inputs, concatenated_cond]


    all_dataset = tf.data.Dataset.from_tensor_slices((concatenated_inputs, concatenated_cond)
    )
    
    total_size = len(all_dataset)  # Replace with your dataset's total size
    print('total size: ',total_size)
    # Define your splitting ratio
    train_size = int(0.8 * total_size)
    test_size = total_size - train_size

    # Create the training dataset
    train_dataset = all_dataset.take(train_size)

    # Create the test dataset
    test_dataset = all_dataset.skip(train_size).take(test_size)
    if eLinks == -1:
        path = os.path.join(args.opath, f'all_eLinks')
    else:
        path = os.path.join(args.opath, f'{eLinks}_eLinks')
    tf.data.experimental.save(train_dataset, path+'_train')
    tf.data.experimental.save(test_dataset, path+'_test')

    
    
args = p.parse_args()
model_dir = args.opath
if not os.path.exists(model_dir):
    os.system("mkdir -p "+model_dir)

for eLinks in [-1]:#[2,3,4,5]:
     
    print(f'Loading {eLinks} eLinks data')
#     model_dir = os.path.join(args.opath, f'{eLinks}_eLinks')
#     if not os.path.exists(model_dir):
#         os.system("mkdir -p " + model_dir)
    
    load_data(eLinks = eLinks)
    

