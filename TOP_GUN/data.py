import torch
from sklearn.metrics import roc_curve
from sklearn import metrics
import os
import einops
import PIL.Image as Image
from torchvision import datasets, transforms

import shutil
import random
import json
import numpy as np
import h5py
import math

from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import datasets, models, transforms
import torchvision.transforms as T

import numpy as np
import matplotlib.pyplot as plt
import timm
import numpy as np
import torch.nn as nn
import torch
from torchvision import datasets, models, transforms
import torchvision
from torchvision.utils import make_grid
from typing import Any
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid
import pyarrow.parquet as pq
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn import metrics
import gc
import os 
import torch
import torch.nn as nn
import sys
import pandas as pd
import wandb
import glob

from einops import rearrange
from einops.layers.torch import Rearrange
import PIL.Image as Image
import PIL as pil
import time
import einops
import pandas as pd
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.utils.data import Dataset

import data
import models

import h5py



def metric(y_true, y_pred):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    auc = metrics.auc(fpr, tpr)
    return auc


# Inspired from https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
class Jet_Dataset(Dataset):
    """Dataset Class"""

    def __init__(self, file_path,chunk_size = 32,MODE = "JET"):
        """
        Arguments:
            file_path (string): Path to the HDF5 file
            chunk_size: The chunk size to read the data from.
            MODE: What to load Jet/Latent
        """
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.MODE = MODE
                
        with h5py.File(self.file_path, 'r') as f:
            self.length = len(f["jet"]) // self.chunk_size

    def __len__(self):
        return self.length
    
    def open_hdf5(self):
        self.file = h5py.File(self.file_path, 'r')
        
    def __getitem__(self, idx: int):
        
        if not hasattr(self, 'file'):
            self.open_hdf5()
        
        # Here idx is the chunk ID
        if self.MODE == "JET":
            data_ = torch.tensor(self.file[f'jet'][idx*self.chunk_size:(idx+1)*self.chunk_size, ...])
        if self.MODE == "LATENT":
            data_ = torch.tensor(self.file[f'latent'][idx*self.chunk_size:(idx+1)*self.chunk_size, ...])
        meta = torch.tensor(self.file[f'meta'][idx*self.chunk_size:(idx+1)*self.chunk_size, ...])

        return data_, meta
        
def collate_with_shuffle(data):
    
    jet = torch.cat([data[i][0] for i in range(len(data))], axis = 0) / 255
    meta = torch.cat([data[i][1] for i in range(len(data))], axis = 0)
    
    indexes = torch.randperm(jet.shape[0])
    jet = jet[indexes]
    meta = meta[indexes]
    return jet, meta

def collate(data):
    
    jet = torch.cat([data[i][0] for i in range(len(data))], axis = 0) / 255
    meta = torch.cat([data[i][1] for i in range(len(data))], axis = 0)
    
    return jet, meta

def collate_with_shuffle_latent(data):
    
    latent = torch.cat([data[i][0] for i in range(len(data))], axis = 0)
    meta = torch.cat([data[i][1] for i in range(len(data))], axis = 0)
    
    indexes = torch.randperm(latent.shape[0])
    latent = latent[indexes]
    latent = torch.flatten(latent,start_dim = 1,end_dim = -1)
    meta = meta[indexes]
    return latent, meta

def collate_latent(data):
    
    latent = torch.cat([data[i][0] for i in range(len(data))], axis = 0)
    meta = torch.cat([data[i][1] for i in range(len(data))], axis = 0)
    latent = torch.flatten(latent,start_dim = 1,end_dim = -1)
    return latent, meta


def populate_latent(ae,jet,latent,BATCH_SIZE = 128,device = "cuda:0"):
    
    NO_SAMPLES = jet.shape[0]
    
    for i in tqdm(range(0,NO_SAMPLES//BATCH_SIZE+1)):
        im = jet[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
        im = torch.tensor(im).float().to(device)
        im = nn.functional.pad(im, (2,1,2,1))
        lat = [ae[j].encoder(im[:,[j],:,:]).to("cpu").squeeze().detach().numpy()[:,None,:] for j in range(8)]
        lat = np.concatenate(lat,axis = 1)
        latent[i*BATCH_SIZE:(i+1)*BATCH_SIZE,:,:] = lat
    print("---------DONE---------")