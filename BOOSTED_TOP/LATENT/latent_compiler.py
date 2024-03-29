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


train_feature = []
train_meta = []
test_feature = []
test_meta = []
for i in tqdm(range(8)):
    train_feature.append(np.load(f"/pscratch/sd/d/diptarko/G23/space/{i}/train_space.npy"))
    test_feature.append(np.load(f"/pscratch/sd/d/diptarko/G23/space/{i}/test_space.npy"))
    
    train_meta.append(pd.read_csv(f"/pscratch/sd/d/diptarko/G23/space/{i}/train_meta.csv"))
    test_meta.append(pd.read_csv(f"/pscratch/sd/d/diptarko/G23/space/{i}/test_meta.csv"))
    
train_feature = np.concatenate(train_feature, axis = -1)
test_feature = np.concatenate(test_feature, axis = -1)

path = f"/pscratch/sd/d/diptarko/G23/space/complete"
if os.path.isdir(path) is False: os.makedirs(path)
np.save(os.path.join(path,"train.npy"),train_feature)
np.save(os.path.join(path,"test.npy"),test_feature)
os.system("cp /pscratch/sd/d/diptarko/G23/space/0/train_meta.csv /pscratch/sd/d/diptarko/G23/space/complete/")
os.system("cp /pscratch/sd/d/diptarko/G23/space/0/test_meta.csv /pscratch/sd/d/diptarko/G23/space/complete/")

print("All done :>")