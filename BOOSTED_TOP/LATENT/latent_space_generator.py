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

from collections import OrderedDict

import torch
import torch.nn as nn
from torch.utils.data import Dataset


    
class depthwise_conv(nn.Module): 
  def __init__(self, nin, kernels_per_layer): 
    super(depthwise_separable_conv, self).__init__() 
    self.depthwise = nn.Conv2d(nin, nin * kernels_per_layer, kernel_size=3, padding=1, groups=nin) 
  
  def forward(self, x): 
    out = self.depthwise(x) 
    return out


    
class auto_encoder(nn.Module):

    def __init__(self, in_channels=3, out_channels=3, init_features=64):
        super(auto_encoder, self).__init__()
        self.in_channels = in_channels

        features = init_features
        self.encoder1 = auto_encoder._block(1, features, name="enc1",groups=self.in_channels)
        # self.pool1 = nn.Conv2d(in_channels=features,out_channels=features,kernel_size=3,stride=2,padding=1)
        self.pool1 = nn.MaxPool2d(2,2)
        
        self.encoder2 = auto_encoder._block(features, features * 2, name="enc2",groups=self.in_channels)
        # self.pool2 = nn.Conv2d(in_channels=features * 2,out_channels=features*2,kernel_size=3,stride=2,padding=1)
        self.pool2 =  nn.MaxPool2d(2,2)
        
        self.encoder3 = auto_encoder._block(features * 2, features * 4, name="enc3",groups=self.in_channels)
        # self.pool3 = nn.Conv2d(in_channels=features * 4,out_channels=features * 4,kernel_size=3,stride=2,padding=1)
        self.pool3 =  nn.MaxPool2d(2,2)
        
        self.encoder4 = auto_encoder._block(features * 4, features * 8, name="enc4",groups=self.in_channels)
        # self.pool4 = nn.Conv2d(in_channels=features * 8,out_channels=features * 8,kernel_size=3,stride=2,padding=1)
        self.pool4 =  nn.MaxPool2d(2,2)
        
        self.encoder5 = auto_encoder._block(features * 8, features * 16, name="enc5",groups=self.in_channels)
        # self.pool5 = nn.Conv2d(in_channels=features * 16,out_channels=features * 16,kernel_size=3,stride=2,padding=1)
        self.pool5 = nn.MaxPool2d(2,2)

        self.bottleneck = auto_encoder._block(features * 16, features * 32, name="bottleneck",groups=self.in_channels)
        self.to_latent = nn.AdaptiveAvgPool2d((1,1))
        self.from_latent =  nn.ConvTranspose2d(features * 32*self.in_channels, features * 32*self.in_channels, kernel_size=4, stride=1,bias=False)
        
        self.upconv5 = nn.ConvTranspose2d(
            features * 32*self.in_channels, features * 16*self.in_channels, kernel_size=2, stride=2,groups=self.in_channels,bias=False)
        self.decoder5 = auto_encoder._block((features * 16), features * 16, name="dec4",groups=self.in_channels)
        
        self.upconv4 = nn.ConvTranspose2d(
            features * 16*self.in_channels, features * 8*self.in_channels, kernel_size=2, stride=2,bias=False
        )
        self.decoder4 = auto_encoder._block((features * 8), features * 8, name="dec4",groups=self.in_channels)
        self.upconv3 = nn.ConvTranspose2d(
            features * 8*self.in_channels, features * 4*self.in_channels, kernel_size=2, stride=2,bias=False
        )
        self.decoder3 = auto_encoder._block((features * 4), features * 4, name="dec3",groups=self.in_channels)
        self.upconv2 = nn.ConvTranspose2d(
            features * 4*self.in_channels, features * 2*self.in_channels, kernel_size=2, stride=2,bias=False
        )
        self.decoder2 = auto_encoder._block((features * 2), features * 2, name="dec2",groups=self.in_channels)
        self.upconv1 = nn.ConvTranspose2d(
            features * 2*self.in_channels, features*self.in_channels, kernel_size=2, stride=2,bias=False
        )
        self.decoder1 = auto_encoder._block(features, features, name="dec1",groups=self.in_channels)

        self.conv = nn.Conv2d(
            in_channels=features*self.in_channels, out_channels=out_channels, kernel_size=1,groups=self.in_channels,bias=False
        )
        
        self.d_stage1 = nn.Sequential(self.encoder1,
                                      self.pool1,
                                      self.encoder2
                                     )
        
        self.d_stage2 = nn.Sequential(self.pool2,
                                      self.encoder3
                                     )

        self.d_stage3 = nn.Sequential(self.pool3,
                                      self.encoder4,
                                     )

        self.d_stage4 = nn.Sequential(self.pool4,
                                      self.encoder5,
                                      )
        
        self.d_stage5 = nn.Sequential(self.pool5,
                                      self.bottleneck)
                                    
        
        self.u_stage5 = nn.Sequential(self.upconv5,
                                      self.decoder5)
        
        self.u_stage4 = nn.Sequential(self.upconv4,
                                      self.decoder4)
        
        self.u_stage3 = nn.Sequential(self.upconv3,
                                      self.decoder3)
        
        self.u_stage2 = nn.Sequential(self.upconv2,
                                      self.decoder2)
        
        self.u_stage1 = nn.Sequential(self.upconv1,
                                      self.decoder1,
                                      self.conv
                                     )

        self.encoder = nn.Sequential(self.d_stage1,
                                     self.d_stage2,
                                     self.d_stage3,
                                     self.d_stage4,
                                     self.d_stage5,
                                     self.to_latent,
                                    )
        self.decoder = nn.Sequential(self.u_stage5,
                                     self.u_stage4,
                                     self.u_stage3,
                                     self.u_stage2,
                                     self.u_stage1,
                                     self.from_latent,
                                    )
        

    def forward(self, x):
        # if mode >= 1:
        x = self.d_stage1(x) #128 --> 64
        # if mode >=2:
        x = self.d_stage2(x) #64 --> 32
        # if mode >=3:
        x = self.d_stage3(x) #32 --> 16
        # if mode >=4:
        x = self.d_stage4(x) #16 --> 8
        # if mode >=5:
        x = self.d_stage5(x) #8 --> 4
        
        # if mode >=6:
        x = self.to_latent(x)
        x = self.from_latent(x)

        # if mode >=5:
        x = self.u_stage5(x)
        # if mode >=4:
        x = self.u_stage4(x)
        # if mode >=3:
        x = self.u_stage3(x)
        # if mode >=2:
        x = self.u_stage2(x)
        # if mode >=1:
        x = self.u_stage1(x)
        return x
    
    # def transit(self,new_mode):
    #     if new_mode > 1:
    #         for para in self.d_stage1.parameters():
    #             para.requires_grad = False
    #         for para in self.u_stage1.parameters():
    #             para.requires_grad = False
    #         print("Froze stage 1")
    #     if new_mode > 2 :
    #         for para in self.d_stage2.parameters():
    #             para.requires_grad = False
    #         for para in self.u_stage2.parameters():
    #             para.requires_grad = False
    #             print("Froze stage 2")
    #     if new_mode > 3 :
    #         for para in self.d_stage3.parameters():
    #             para.requires_grad = False
    #         for para in self.u_stage3.parameters():
    #             para.requires_grad = False
    #             print("Froze stage 3")
    #     if new_mode > 4 :
    #         for para in self.d_stage4.parameters():
    #             para.requires_grad = False
    #         for para in self.u_stage4.parameters():
    #             para.requires_grad = False
    #             print("Froze stage 4")
    #     if new_mode > 5 :
    #         for para in self.d_stage5.parameters():
    #             para.requires_grad = False
    #         for para in self.u_stage5.parameters():
    #             para.requires_grad = False
    #             print("Froze stage 5")



    # @staticmethod
    def _block(in_channels, features, name, groups):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels*groups,
                            out_channels=features*groups,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                            groups=groups,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features*groups)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                ]
            )
        )

    
class restruct(torch.nn.Module):
    
    def __init__(self,idx):
        super().__init__()
        self.idx = idx
        
    def forward(self, img):
        """
        Args:
            img (Tensor): The stacked Image .
        Returns:
            Tensor: Restructured Image into 8 channels.
        """   
        return einops.rearrange(torch.squeeze(img), 'h ( w c ) -> c h w ', w = 125, c=8)[idx,:,:].unsqueeze(0)
    
class ToTensor_(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, img):
        return transforms.functional.to_tensor(img)
    

def pil_loader(path: str):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('L')

class DirImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_list = os.listdir(img_dir)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_list[idx])
        image = pil_loader(img_path)
        
        mass = float(self.img_list[idx].split("_")[2])
        pt = float(self.img_list[idx].split("_")[4])
        y = int(self.img_list[idx].split("_")[6][0])
        
        if self.transform:
            image = self.transform(image)
        return image, mass, pt, y    
    
    
def space(dataloader,name,device):
    latent_space = []
    mass_list = []
    pt_list = []
    y_list = []

    with torch.no_grad():
        for image, mass, pt, y in tqdm(dataloader):
            image = image.to(device)
            with torch.no_grad():
                image = nn.functional.pad(image, (2,1,2,1))
                latent = model.encoder(image)
                latent = latent.squeeze()
                latent_space.extend(latent.detach().cpu().numpy())
                mass_list.extend(mass.numpy())
                pt_list.extend(pt.numpy())
                y_list.extend(y.numpy())
    latent_space = np.array(latent_space)

    path = f"/pscratch/sd/d/diptarko/G23/space/{idx}"
    if os.path.isdir(path) is False: os.makedirs(path)

    df = pd.DataFrame()
    df["Mass"] = mass_list
    df["pt"] =  pt_list
    df["y"] = y_list
    df.to_csv(os.path.join(path,f"{name}_meta.csv"))
    np.save(os.path.join(path,f"{name}_space.npy"),latent_space)


    
if __name__ == "__main__":
    idx = int(sys.argv[1])
    device = int(sys.argv[2])
    device = f"cuda:{device}"

    train_transform = transforms.Compose([
                            ToTensor_(),
                            restruct(idx),
                            # Train_transform()
    ])



    test_transform = transforms.Compose([
                                ToTensor_(),
                                restruct(idx)])

    dataset_train = DirImageDataset("/pscratch/sd/d/diptarko/Top_full/Train/Train/",
                                         transform =train_transform)

    dataset_test = DirImageDataset("/pscratch/sd/d/diptarko/Top_full/Test/Test/",
                                        transform =test_transform)




    dataloader_test = torch.utils.data.DataLoader(dataset_test,
                                                batch_size=512,
                                                shuffle=False,
                                                drop_last = False,
                                                num_workers=8,
                                                pin_memory = True)

    dataloader_train = torch.utils.data.DataLoader(dataset_train,
                                                batch_size=512,
                                                shuffle=False,
                                                drop_last = False,
                                                num_workers=8,
                                                pin_memory = True)
    
    PATH = f"/pscratch/sd/d/diptarko/dc250601/latent/{idx}/model_Epoch_{10}.pt"
    checkpoint = torch.load(PATH)
    model = auto_encoder(in_channels=1,out_channels=1, init_features=32)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    print()
    
    space(dataloader_train,"train",device)
    space(dataloader_test,"test",device)