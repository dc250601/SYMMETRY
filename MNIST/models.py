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


#--------------------------------------------------------------------------------------
#                            TANGENTIAL FLOW BASED MODELS
#--------------------------------------------------------------------------------------

class UNET(nn.Module):
    """
    UNET: Function to create an UNET based model. Input/Ouput is in range of (-1,1).
    image_channels (int): The number of channels of the input and output image
    init_channels (int): The init_feature size, with default value of 8, increasing this increases expressivity
    """
    def __init__(self,image_channels = 1,init_channels = 8):
        super(UNET, self).__init__()
 
    # encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(
                in_channels=image_channels,
                out_channels=init_channels,
                kernel_size=4, 
                stride=2,
                padding=1),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=init_channels,
                out_channels=init_channels,
                kernel_size=3, 
                stride=1,
                padding=1),
            nn.ReLU())
                                  # 28 ---> 14
        self.enc2 = nn.Sequential(
            nn.Conv2d(
                in_channels=init_channels,
                out_channels=init_channels*2,
                kernel_size=4, 
                stride=2,
                padding=1),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=init_channels*2,
                out_channels=init_channels*2,
                kernel_size=3, 
                stride=1,
                padding=1),
            nn.ReLU())
        # 14 ---> 7
        self.enc3 = nn.Sequential(
            nn.Conv2d(
                in_channels=init_channels*2,
                out_channels=init_channels*4,
                kernel_size=3, 
                stride=2,
                padding=1),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=init_channels*4,
                out_channels=init_channels*4,
                kernel_size=3, 
                stride=1,
                padding=1),
            nn.ReLU())
            # 7 ---> 4
        
        # decoder 
        self.upscale1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=init_channels*4,
                out_channels=init_channels*2,
                kernel_size=3, 
                stride=2,
                padding=1)
        ) #Has to be 4 ---> 7 not yet fixed !!!
        self.dec1 = nn.Sequential(
            nn.Conv2d(
                in_channels=init_channels*4, # Upscale Bottleneck 2 + Skip (4) = 8
                out_channels=init_channels*2,
                kernel_size=3, 
                stride=1,
                padding=1),
            nn.ReLU())
        
        self.upscale2 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=init_channels*2,
                out_channels=init_channels*1,
                kernel_size=2, 
                stride=2,
                padding=0)
        ) #Has to be 7 ---> 14 
        
        self.dec2 = nn.Sequential(
            nn.Conv2d(
                in_channels=init_channels*2, # Upscale2(2) + Skip enc2(2) = 4
                out_channels=init_channels*1,
                kernel_size=3, 
                stride=1,
                padding=1),
            nn.ReLU())

        self.upscale3 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=init_channels,
                out_channels=init_channels,
                kernel_size=2, 
                stride=2,
                padding=0)
        ) #Has to be 7 ---> 14 
        
        self.dec3 = nn.Sequential(
            nn.Conv2d(
                in_channels=init_channels, # Upscale3(1) + Skip enc1(1) = 2
                out_channels=init_channels,
                kernel_size=3, 
                stride=1,
                padding=1),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=init_channels, # Upscale3(1) + Skip enc1(1) = 2
                out_channels=image_channels,
                kernel_size=3, 
                stride=1,
                padding=1)
        ) #Has to be 14 ---> 28


 
    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x2 = self.dec1(torch.concat([x2,self.upscale1(x3)], dim=1))
        x1 = self.dec2(torch.concat([x1,self.upscale2(x2)], dim=1))
        x  =  nn.Tanh()(self.dec3(torch.concat([self.upscale3(x1)], dim=1)))
        
        return x
    
#--------------------------------------------------------------------------------------


class ImageOracle(nn.Module):
    def __init__(self, image_channels = 1, init_channels = 8):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(
                in_channels=image_channels,
                out_channels=init_channels,
                kernel_size=4, 
                stride=2,
                padding=1),
            nn.ReLU())
                                  # 28 ---> 14
        self.enc2 = nn.Sequential(
            nn.Conv2d(
                in_channels=init_channels,
                out_channels=init_channels*2,
                kernel_size=4, 
                stride=2,
                padding=1),
            nn.ReLU())
        # 14 ---> 7
        self.enc3 = nn.Sequential(
            nn.Conv2d(
                in_channels=init_channels*2,
                out_channels=init_channels*4,
                kernel_size=3, 
                stride=2,
                padding=1),
            nn.ReLU())
            # 7 ---> 4
        self.adapt = nn.AdaptiveAvgPool2d((1,1))
        self.relu = nn.ReLU()
        self.head = nn.Linear(init_channels*4,9)
        

    def forward(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.relu(self.adapt(x)).squeeze()
        x = self.head(x).squeeze()
        return x

#--------------------------------------------------------------------------------------


class ImageDescriminator(nn.Module):
    def __init__(self, image_channels = 1, init_channels = 8):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(
                in_channels=image_channels,
                out_channels=init_channels,
                kernel_size=4, 
                stride=2,
                padding=1),
            nn.ReLU())
                                  # 28 ---> 14
        self.enc2 = nn.Sequential(
            nn.Conv2d(
                in_channels=init_channels,
                out_channels=init_channels*2,
                kernel_size=4, 
                stride=2,
                padding=1),
            nn.ReLU())
        # 14 ---> 7
        self.enc3 = nn.Sequential(
            nn.Conv2d(
                in_channels=init_channels*2,
                out_channels=init_channels*4,
                kernel_size=3, 
                stride=2,
                padding=1),
            nn.ReLU())
            # 7 ---> 4
        self.adapt = nn.AdaptiveAvgPool2d((1,1))
        self.relu = nn.ReLU()
        self.head = nn.Linear(init_channels*4,1) # 1 output feature since it will differetiate FAKES from REALS
        

    def forward(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.relu(self.adapt(x)).squeeze()
        x = self.head(x).squeeze()
        return x

#--------------------------------------------------------------------------------------

class ImageGenerator(nn.Module):
    def __init__(self, init_channels = 8):
        super().__init__()    
        self.map = UNET(init_channels=init_channels)

    def forward(self,x,theta, order = 10):

        nom = x
        result = x
        denom = 1.0
        # theta = (2*torch.rand(x.shape[0]) - 1)[:,None,None,None].expand(x.shape) Generate theta using this form while training
        for i in range(1, order):
            nom = (self.map(nom))
            denom *= i
            result = result + ((theta)**i)*(nom/denom)
        return result
#--------------------------------------------------------------------------------------
#                            LATENT FLOW BASED MODELS
#--------------------------------------------------------------------------------------

class ConvVAE(nn.Module):
    def __init__(self,
                 init_channels = 8,
                 image_channels = 1,
                 kernel_size = 4,
                 latent_dim = 16):
        
        super(ConvVAE, self).__init__()
 
        # encoder
        self.enc1 = nn.Conv2d(
            in_channels=image_channels, out_channels=init_channels, kernel_size=kernel_size, 
            stride=2, padding=1
        )
        self.enc2 = nn.Conv2d(
            in_channels=init_channels, out_channels=init_channels*2, kernel_size=kernel_size, 
            stride=2, padding=1
        )
        self.enc3 = nn.Conv2d(
            in_channels=init_channels*2, out_channels=init_channels*4, kernel_size=kernel_size, 
            stride=2, padding=2
        )
        self.enc4 = nn.Conv2d(
            in_channels=init_channels*4, out_channels=64, kernel_size=kernel_size, 
            stride=2, padding=0
        )
        # fully connected layers for learning representations
        self.fc1 = nn.Linear(64, 128)
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_log_var = nn.Linear(128, latent_dim)
        self.fc2 = nn.Linear(latent_dim, 64)
        # decoder 
        self.dec1 = nn.ConvTranspose2d(
            in_channels=64, out_channels=init_channels*8, kernel_size=kernel_size, 
            stride=1, padding=0
        )
        self.dec2 = nn.ConvTranspose2d(
            in_channels=init_channels*8, out_channels=init_channels*4, kernel_size=kernel_size, 
            stride=2, padding=2
        )
        self.dec3 = nn.ConvTranspose2d(
            in_channels=init_channels*4, out_channels=init_channels*2, kernel_size=kernel_size, 
            stride=2, padding=0
        )
        self.dec4 = nn.ConvTranspose2d(
            in_channels=init_channels*2, out_channels=image_channels, kernel_size=kernel_size, 
            stride=2, padding=1
        )
    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        sample = mu + (eps * std) # sampling
        return sample
    
    def encoder(self, x):
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        x = F.relu(self.enc3(x))
        x = F.relu(self.enc4(x))
        batch, _, _, _ = x.shape
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch, -1)
        hidden = self.fc1(x)
        
        return hidden
    
    def decoder(self, z):
        x = F.relu(self.dec1(z))
        x = F.relu(self.dec2(x))
        x = F.relu(self.dec3(x))
        reconstruction = torch.sigmoid(self.dec4(x))
        
        return reconstruction
 
    def forward(self, x):
        # encoding
        hidden = self.encoder(x)
        # get `mu` and `log_var`
        mu = self.fc_mu(hidden)
        log_var = self.fc_log_var(hidden)
        # get the latent vector through reparameterization
        z = self.reparameterize(mu, log_var)
        z = self.fc2(z)
        z = z.view(-1, 64, 1, 1)
 
        # decoding
        reconstruction = self.decoder(z)
        
        return reconstruction, mu, log_var

#--------------------------------------------------------------------------------------

class AutoEncoder(nn.Module):
    def __init__(self, feature_multipier = 1):
        super().__init__()
        mul = feature_multipier

        num_features = 4
        self.encoder = nn.Sequential(
                                nn.Conv2d(in_channels=1,
                                          out_channels=num_features*mul,
                                          kernel_size=3,
                                          bias=False,
                                          padding=3),
                                nn.BatchNorm2d(num_features*mul),
                                nn.ReLU(inplace=True),
        
                                nn.MaxPool2d(2,2), # 32 --> 16
                                
                                nn.Conv2d(in_channels=num_features*mul,
                                          out_channels=num_features*mul*2,
                                          kernel_size=3,
                                          bias=False,
                                          padding=1),
                                nn.BatchNorm2d(num_features*mul*2),
                                nn.ReLU(inplace=True),
        
                                nn.MaxPool2d(2,2), # 16 --> 8
        
                                nn.Conv2d(in_channels=num_features*mul*2,
                                          out_channels=num_features*mul*4,
                                          kernel_size=3,
                                          bias=False,
                                          padding=1),
                                nn.BatchNorm2d(num_features*mul*4),
                                nn.ReLU(inplace=True),
        
                                nn.MaxPool2d(2,2), # 8 --> 4
        
                                nn.Conv2d(in_channels=num_features*mul*4,
                                          out_channels=num_features*mul*8,
                                          kernel_size=3,
                                          bias=False,
                                          padding=1),
                                nn.BatchNorm2d(num_features*mul*8),
                                nn.ReLU(inplace=True),
        
                                nn.AdaptiveAvgPool2d((1,1))
                               )
        
        self.decoder = nn.Sequential(
                                nn.ConvTranspose2d(in_channels=num_features*mul*8,
                                                   out_channels=num_features*mul*8,
                                                   kernel_size=4,
                                                   stride=1),
                                nn.Conv2d(in_channels=num_features*mul*8,
                                          out_channels=num_features*mul*8,
                                          kernel_size=3,
                                          bias=False,
                                          padding=1),
                                nn.BatchNorm2d(num_features*mul*8),
                                nn.ReLU(inplace=True),   # 1X1 --> 3X3
        
                                nn.ConvTranspose2d(in_channels=num_features*mul*8,
                                                   out_channels=num_features*mul*4,
                                                   kernel_size=2,
                                                   stride=2),
                                nn.Conv2d(in_channels=num_features*mul*4,
                                          out_channels=num_features*mul*4,
                                          kernel_size=3,
                                          bias=False,
                                          padding=1), #4 --> 8
                                nn.BatchNorm2d(num_features*mul*4),
                                nn.ReLU(inplace=True),
        
            
                                nn.ConvTranspose2d(in_channels=num_features*mul*4,
                                                   out_channels=num_features*mul*2,
                                                   kernel_size=2,
                                                   stride=2),
                                nn.Conv2d(in_channels=num_features*mul*2,
                                          out_channels=num_features*mul*2,
                                          kernel_size=3,
                                          bias=False,
                                          padding=0), #8 --> 14
                                nn.BatchNorm2d(num_features*mul*2),
                                nn.ReLU(inplace=True),
        
            
                                nn.ConvTranspose2d(in_channels=num_features*mul*2,
                                                   out_channels=num_features*mul*1,
                                                   kernel_size=2,
                                                   stride=2),
                                nn.Conv2d(in_channels=num_features*mul*1,
                                          out_channels=1,
                                          kernel_size=3,
                                          bias=False,
                                          padding=1), #14X14 --> 28X28
                                 )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x

#--------------------------------------------------------------------------------------

class LatentOracle(nn.Module):
    def __init__(self, feature_multipier = 1):
        super().__init__()
        self.mul = feature_multipier
        self.layer1 = nn.LazyLinear(16*feature_multipier)
        self.bn1 = nn.LazyBatchNorm1d()
        self.relu = nn.ReLU()
        self.head = nn.LazyLinear(9)
        

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
