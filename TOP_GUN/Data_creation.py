import numpy as np
import torch
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import os
import sys
import glob
import h5py
import math

FILE_LIST = glob.glob("/pscratch/sd/d/diptarko/SYMMETRY/TOP_GUN/assets/DATA/Data_AM/*.parquet")
NO_SAMPLES = sum(list(pq.ParquetFile(file).metadata.num_rows for file in FILE_LIST))

data = h5py.File('/pscratch/sd/d/diptarko/SYMMETRY/TOP_GUN/assets/DATA/Regression_AM.h5', 'w')

data_chunk_meta = data.create_dataset("meta", (NO_SAMPLES,4))
data_chunk_jet = data.create_dataset("jet", (NO_SAMPLES,8,125,125))

index = 0
for file in tqdm(FILE_LIST):
    file = pq.ParquetFile(file)
    batch_iter = file.iter_batches(batch_size=4096,use_threads=True)
    
    while(True):
        try:
            batch = next(batch_iter)
            try:
                
                p = batch.to_pandas(use_threads=True)
                samples = p.shape[0]
                im = np.array(np.array(np.array(p.iloc[:,0].tolist()).tolist()).tolist())
                meta = np.array(p.iloc[:,1:])
                data_chunk_meta[index:(index+samples),:] = meta
                data_chunk_jet[index:(index+samples),:,:,:] = im
                index += samples
                
                
            except:
                print("Something went wrong with the IO")
        except:
            break
            
data.close()
data = h5py.File('/pscratch/sd/d/diptarko/SYMMETRY/TOP_GUN/assets/DATA/Regression_AM.h5', 'r+')
BATCH_SIZE = 4096
CHANNELS = 8

mean_ = []
std_ = []
print
for i in tqdm(range(math.ceil(data["jet"].shape[0]/BATCH_SIZE))):
    im = data["jet"][i*BATCH_SIZE:(i+1)*BATCH_SIZE,:,:,:]
    im[im < 1.e-3] = 0 #Zero_suppression
    mean_.append(list(im[:,j,:,:].mean() for j in range(CHANNELS)))
    std_.append(list(im[:,j,:,:].std() for j in range(CHANNELS)))
    

mean = np.array(mean_).mean(axis = 0)
std = ((np.array(std_)**2).mean(axis = 0)+np.array(mean_).std(axis = 0)**2)**0.5

for i in tqdm(range(data["jet"].shape[0])):
    im = data["jet"][i,:,:,:]
    im[im < 1.e-3] = 0 #Zero_suppression
    for j in range(CHANNELS):
        im[j,:,:] = (im[j,:,:] - mean[j])/(std[j])
        im[j,:,:] = np.clip(im[j,:,:], 0, 500*im[j,:,:].std())
        im[j,:,:] = 255*(im[j,:,:])/(im[j,:,:].max())
    im = im.astype(np.uint8)
    data["jet"][i,:,:,:] = im

data.close()