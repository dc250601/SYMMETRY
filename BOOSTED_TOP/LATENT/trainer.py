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
# import pandas as pd
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
import pandas as pd
import sys

Batch_Size = 512
n_epochs = 100

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

class Train_transform(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.hflip = transforms.RandomHorizontalFlip()
        self.rflip = transforms.RandomVerticalFlip()
        self.rr = transforms.RandomRotation(60)
    
    def forward(self, img):
        img = self.hflip(img)
        img = self.rflip(img)
        img = self.rr(img)
        
        return img
    
    
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
    
    
if __name__ == "__main__":
    idx = int(sys.argv[1])
    device = int(sys.argv[2])
    device = f"cuda:{device}"

    path = "/pscratch/sd/d/diptarko/dc250601/registry/"
    Total_No_epochs = int(sys.argv[3])
    Present_Lr = 1e-3
    FEATURES  = int(sys.argv[4])
    Run_Name = f"Boosted_Top_Channel_{idx}_size_{FEATURES}"
    new_ = 1
    Present_Epoch = 0 
    if os.path.isfile(os.path.join(path,str(Run_Name)+".csv")):
        print("File Exits")
        df = pd.read_csv(os.path.join(path,str(Run_Name)+".csv"))
        Present_Lr = df["Present_Lr"][0]
        Present_Epoch = df["Present_Epoch"][0]
        Run_Id = df["Run_Id"][0]
        Run_Name = df["Run_Name"][0]
        new_ = df["New"][0]
    else:
        df = pd.DataFrame()
        df["Total_No_epochs"] = [Total_No_epochs]
        df["Present_Lr"] = [Present_Lr]
        df["Present_Epoch"] = [Present_Epoch]
        df["Run_Id"] = [wandb.util.generate_id()]
        Run_Id = df["Run_Id"][0]
        df["New"] = [1]
        df["Run_Name"] = [Run_Name]
        df.to_csv(os.path.join(path,str(Run_Name)+".csv"))
        os.mkdir(f"/pscratch/sd/d/diptarko/dc250601/{Run_Name}")
        print("File Created")
    
    if Present_Lr < 1e-8:
        sys.exit("LR too LOW ----")

    
    train_transform = transforms.Compose([
                            ToTensor_(),
                            restruct(idx),
                            # Train_transform()
    ])



    test_transform = transforms.Compose([
                                ToTensor_(),
                                restruct(idx)])
        
    dataset_train = datasets.ImageFolder("/dev/shm/Top_full/Train",
                                         transform =train_transform,
                                        loader = pil_loader)

    dataset_test = datasets.ImageFolder("/dev/shm/Top_full/Test/",
                                        transform =test_transform,
                                        loader = pil_loader)




    dataloader_test = torch.utils.data.DataLoader(dataset_test,
                                                batch_size=Batch_Size,
                                                shuffle=True,
                                                drop_last = True,
                                                num_workers=4,
                                                persistent_workers = True,
                                                pin_memory = True)

    dataloader_train = torch.utils.data.DataLoader(dataset_train,
                                                batch_size=Batch_Size,
                                                shuffle=True,
                                                drop_last = True,
                                                num_workers=12,
                                                persistent_workers = True,
                                                pin_memory = True)



    device = device
    model = auto_encoder(in_channels=1,out_channels=1, init_features=FEATURES)

    if new_ == 0:

        checkpoint = torch.load(f"/pscratch/sd/d/diptarko/dc250601/{Run_Name}/model_Epoch_{Present_Epoch-1}.pt")
        model.load_state_dict(checkpoint['model_state_dict'])

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(),lr = Present_Lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose = True,threshold = 5e-2, patience = 5, factor = 0.1)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    wandb.login(key="410945fd39931e71a60b7d626e541f6988672b95")
    wandb.init(
        project = "SYMMETRY_Latent",
        name = Run_Name,
        id = Run_Id,
        resume = "allow"
    )
    

    last_loss = 100
    for epoch in range(Present_Epoch,n_epochs,1):
        train_loss = 0
        val_loss = 0
        train_steps = 0
        test_steps = 0

        val_loss = 0

        model.train()
        for image, _ in dataloader_train:
            image = image.to(device)
            with torch.no_grad():
                image = nn.functional.pad(image, (2,1,2,1))

            optimizer.zero_grad()

            
            outputs = model(image)
            loss = criterion(outputs, image)
                
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_steps += 1


        #-------------------------------------------------------------------
        model.eval()
        with torch.no_grad():
            for image, _ in dataloader_test:
                image = image.to(device)

                image = nn.functional.pad(image, (2,1,2,1))

                outputs = model(image)
                loss = criterion(outputs, image)
                val_loss += loss.item()
                test_steps +=1


        train_loss = train_loss/train_steps
        val_loss = val_loss/ test_steps

        print("----------------------------------------------------")
        print("Epoch No" , epoch)
        print("The Training loss of the epoch, ",train_loss)
        print("The validation loss of the epoch, ",val_loss)
        print("----------------------------------------------------")

        grid = make_grid(make_grid(list(image[0,i,:,:].cpu() for i in range(1)) + list(nn.Sigmoid()(outputs[0,i,:,:].detach().cpu()) for i in range(1))).unsqueeze(1))
        log_img = torchvision.transforms.ToPILImage()(grid)

        scheduler.step((last_loss-val_loss)/last_loss)
        last_loss = val_loss
        curr_lr = scheduler._last_lr[0]
        wandb.log({"Epoch": epoch,
                    "Train_loss_epoch": train_loss,
                    "Val_loss_epoch": val_loss,
                    "Lr": curr_lr,
                    "Embedding":wandb.Image(log_img),
                   # "stage":stage
                  }
                    )
        PATH = f"/pscratch/sd/d/diptarko/dc250601/{Run_Name}/model_Epoch_{epoch}.pt"
        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
                }, PATH)
        df = pd.read_csv(os.path.join(path,str(Run_Name)+".csv"))
        df.loc["Present_Lr"][0] = curr_lr
        df.loc["Present_Epoch"][0] = epoch + 1
        dfloc["New"] = [0]
        df.to_csv(os.path.join(path,str(Run_Name)+".csv"))
        
        if curr_lr < 1e-8:
            sys.exit()
            
        gc.collect()

    wandb.finish()