import sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import CenterCrop

import lightning as L
import einops
from einops.layers.torch import Rearrange


#############################
class CGenerator(nn.Module):
    def __init__(self, num_classes=10, nz=5):
        super().__init__()

        self.nz = nz
        self.num_classes = num_classes


        # ConvNet Output Size Calculator https://asiltureli.github.io/Convolution-Layer-Calculator/
        self.z_project = nn.Linear(nz, 50, bias=False)   
        self.label_emb = nn.Embedding(num_classes, 50)
        self.project = nn.Linear(100, 256 * 7 * 7, bias=False)

        self.main = nn.Sequential(
            nn.BatchNorm1d(256 * 7 * 7),
            nn.LeakyReLU(),
            Rearrange('b (c h w) -> b c h w', c=256, h=7, w=7),  

            nn.ConvTranspose2d(256, 128, kernel_size=5, stride=1, padding=2, bias=False),  
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),  

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False), 
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=2, stride=1, padding=1),  

            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1, bias=False),    
            nn.Tanh(),

            CenterCrop(28)  
        )

        
    
    def forward(self, z, c):
        z_emb = self.z_project(z)
        c_emb = self.label_emb(c)  # (B, 50)
        x = torch.cat([z_emb, c_emb], dim=1)  # (B, nz+50)
        x = self.project(x)
        x = self.main(x)
        return x



class CDiscriminator(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.num_classes = num_classes

        # ConvNet Output Size Calculator https://asiltureli.github.io/Convolution-Layer-Calculator/
        self.label_emb = nn.Embedding(num_classes, 28 * 28)

        self.main = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1, bias=False),
            nn.Sigmoid(),
        )
        
    def forward(self, x, c):
        c_emb = self.label_emb(c).view(-1, 1, 28, 28)
        x = torch.cat([x, c_emb], dim=1)  
        return self.main(x)
    


class CDCGAN(L.LightningModule):
    def __init__(self, num_classes=10, nz=5, lr=0.0002, beta1=0.5):
        super().__init__()
        self.automatic_optimization = False

        self.num_classes = num_classes
        self.nz = nz
        self.lr = lr
        self.beta1 = beta1
        self.save_hyperparameters()

        self.generator = CGenerator(nz=self.nz, num_classes=self.num_classes)
        self.discriminator = CDiscriminator(num_classes=self.num_classes)

    def forward(self, z, c):
        return  self.generator(z,c)
    
    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)
    
    def training_step(self, batch, batch_idx):
        x, c = batch
        batch_size = x.size(0)

        opt_g, opt_d = self.optimizers()

        # sample noises
        z = torch.randn(batch_size, self.nz, device=self.device)

        # generator_step
        self.toggle_optimizer(opt_g)
        fake = self(z, c)  # 생성된 이미지
        pred_fake = self.discriminator(fake, c)

        loss_G = self.adversarial_loss(pred_fake, torch.ones_like(pred_fake))
        self.log("loss_G", loss_G, prog_bar=True)
        self.manual_backward(loss_G)
        opt_g.step()
        opt_g.zero_grad()
        self.untoggle_optimizer(opt_g)

        # discriminator_step
        self.toggle_optimizer(opt_d)
        z = torch.randn(batch_size, self.nz, device=self.device)
        fake = self(z, c).detach()

        pred_real = self.discriminator(x, c)
        pred_fake = self.discriminator(fake, c)

        loss_D_real = self.adversarial_loss(pred_real, torch.ones_like(pred_real))
        loss_D_fake = self.adversarial_loss(pred_fake, torch.zeros_like(pred_fake))
        loss_D = (loss_D_real + loss_D_fake) / 2

        self.log("loss_D", loss_D, prog_bar=True)
        self.manual_backward(loss_D)
        opt_d.step()
        opt_d.zero_grad()
        self.untoggle_optimizer(opt_d)


    def configure_optimizers(self):
        opt_G = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        opt_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        return [opt_G, opt_D], []

