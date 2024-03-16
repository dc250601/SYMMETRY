import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from torch.utils.data import TensorDataset, DataLoader
from torch.distributions.normal import Normal
import torch.nn.functional as F
import gc
import imutils
import math

def get_dataset_augmented():
    with open('./augmented_dataset.data', 'rb') as f:
        dataset = pickle.load(f)

    X_train = dataset["X_train"]
    Y_train = dataset["Y_train"]

    X_test = dataset["X_test"]
    Y_test = dataset["Y_test"]

    X_train = np.array(X_train)/255
    X_test = np.array(X_test)/255
    Y_train = np.array(Y_train)
    Y_test = np.array(Y_test)

    X_train = torch.Tensor(X_train[:,None,:,:])
    X_test = torch.Tensor(X_test[:,None,:,:])

    Y_train = torch.Tensor(Y_train)
    Y_test = torch.Tensor(Y_test)
    
    return X_train, X_test, Y_train, Y_test

def get_dataset_distilled():
    
    X_train_ = torch.tensor(np.load("./dist_train_x.npy"))
    X_test_ = torch.tensor(np.load("./dist_test_x.npy"))

    Y_train_ = torch.tensor(np.load("./dist_train_y.npy"))
    Y_test_ = torch.tensor(np.load("./dist_test_y.npy"))

    return X_train, X_test, Y_train, Y_test