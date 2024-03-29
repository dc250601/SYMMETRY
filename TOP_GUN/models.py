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
from collections import OrderedDict



#--------------------------------------------------------------------------------------
#                            LATENT FLOW BASED MODELS
#--------------------------------------------------------------------------------------

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


#--------------------------------------------------------------------------------------

class LatentOracle(nn.Module):
    def __init__(self, feature_multipier = 1):
        super().__init__()
        self.mul = feature_multipier
        self.layer1 = nn.LazyLinear(16*feature_multipier)
        self.bn1 = nn.LazyBatchNorm1d()
        self.relu = nn.ReLU()
        self.head = nn.LazyLinear(1)
        

    def forward(self, x):
        x = self.layer1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.head(x)
        return x
    
#--------------------------------------------------------------------------------------

class LatentDescriminator(nn.Module):
    def __init__(self, feature_multipier = 1):
        super().__init__()
        self.mul = feature_multipier
        self.layer1 = nn.LazyLinear(16*feature_multipier)
        self.bn1 = nn.LazyBatchNorm1d()
        self.relu = nn.LeakyReLU()
        self.head = nn.LazyLinear(1)
        

    def forward(self, x):
        x = self.layer1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.head(x)
        return x
    
       
#--------------------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self,feature_size, feature_multipier = 1):
        super().__init__()
        
        self.layer1 = nn.Linear(feature_size,feature_size*feature_multipier)
        # self.bn1 = nn.BatchNorm1d(64*feature_multipier) #Commenting for now will remove later to increases expressivity
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(feature_size*feature_multipier,feature_size)
        

    def forward(self, x):
        x = self.layer1(x)
        # x = self.bn1(x) #Commenting for now will remove later to increase expressivity
        x = self.relu(x)
        
        x = self.layer2(x)
        return x

#--------------------------------------------------------------------------------------

class MLP_Tanh(nn.Module):
    def __init__(self,feature_size, feature_multipier = 1):
        super().__init__()
        
        self.layer1 = nn.Linear(feature_size,feature_size*feature_multipier)
        # self.bn1 = nn.BatchNorm1d(64*feature_multipier) #Commenting for now will remove later to increases expressivity
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(feature_size*feature_multipier,feature_size)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.layer1(x)
        # x = self.bn1(x) #Commenting for now will remove later to increase expressivity
        x = self.relu(x)
        
        x = self.layer2(x)
        x = self.tanh(x)  
        return x
    
#--------------------------------------------------------------------------------------

class space(nn.Module):
    def __init__(self,feature_size,feature_multipier = 1):
        super().__init__()
        
        self.encoder = MLP(feature_size = feature_size, feature_multipier = feature_multipier)
        self.decoder = MLP(feature_size = feature_size, feature_multipier = feature_multipier)
        
    def forward(self,x):
        print("ERROR:302 NOT INTENDED TO BE USED IN THIS WAY")
        return self.decoder(self.encoder(x))

#--------------------------------------------------------------------------------------


class GeneratorLatent(nn.Module):
    def __init__(self, num_features):
        super().__init__()    
        self.num_features = num_features
        self.algebra = torch.nn.Parameter(torch.empty((num_features,num_features)))
        self.reset_parameters()
        
    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.algebra, a=math.sqrt(5))
    
    def action(self,x):
        return torch.mm(x,self.algebra)

    
#--------------------------------------------------------------------------------------

class GroupLatent(nn.Module):
    def __init__(self,num_features, num_generators, LOSS_MODE = "MAE"): ## MAE works better than MSE but takes longer to converge and gives sparser generators
        super().__init__()    
        self.num_generators = num_generators
        self.LOSS_MODE = LOSS_MODE
        self.num_features = num_features
        self.group = nn.ModuleList([GeneratorLatent(self.num_features) for i in range(self.num_generators)])
        self.reset_parameters()
        self.criterion_cos = nn.CosineSimilarity(dim=0)
        
    def reset_parameters(self) -> None:
        for generator in self.group:
            generator.reset_parameters()
    
    def forward(self, theta, x, order = 10):
        t = x
        result = x
        for i in range(1,order+1):
            z = 0
            for j, generator in enumerate(self.group):
                z = z + (theta[j][:,None]/i) * generator.action(t)
            result = result + z
            t = z
        return result
    
    def collapse_loss(self):
        
        loss = 0
        zero = torch.zeros((self.num_features,self.num_features),device = self.group[0].algebra.device)

        for generator in self.group:
            if self.LOSS_MODE == "MAE":
                loss = loss + torch.mean(torch.abs(self.criterion_cos(zero,generator.algebra)))
            if self.LOSS_MODE == "MSE":
                loss = loss + torch.mean(self.criterion_cos(zero,generator.algebra)**2)
        
        return loss
    
    def orthogonal_loss(self):
        
        loss = 0
        
        for i,generator1 in enumerate(self.group):
            for j,generator2 in enumerate(self.group):
                if i!=j:
                    if self.LOSS_MODE == "MAE":
                        loss = loss + torch.mean(torch.abs(self.criterion_cos(generator1.algebra,generator2.algebra)))
                    if self.LOSS_MODE == "MSE":
                        loss = loss + torch.mean(self.criterion_cos(generator1.algebra,generator2.algebra)**2)
        
        return loss/2
        
#--------------------------------------------------------------------------------------
