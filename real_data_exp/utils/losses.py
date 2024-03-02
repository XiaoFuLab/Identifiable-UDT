from typing import Union
from typing_extensions import Literal
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class GANLoss(nn.Module):
    def __init__(self, type=None, device='cpu'):
        super(GANLoss, self).__init__()
        if type is None:
            self.type = 'lsgan'
        else:
            self.type = type
        losses = {'lsgan': self.LSGAN_loss, 
                  'nsgan': self.NSGAN_loss,
                  'label_smooth_gan': self.LabelSmoothGAN_loss}
        
        self.loss = losses[self.type]
        self.bce_logit_loss = nn.BCEWithLogitsLoss()
        self.device = device

    def forward(self, fake, real=None, is_disc=False, weight=None):
        return self.loss(fake, real, is_disc, weight=weight)
    
    def LSGAN_loss(self, fake, real=None, is_disc=False, weight=None):
        if is_disc:
            assert real is not None, 'Discriminator Loss: real is None'
            w = weight if weight is not None else torch.ones_like(fake).to(self.device)
            return torch.mean((real-1)**2) + torch.mean(w * (fake**2))
        else:
            return torch.mean((fake-1)**2)
    
    def NSGAN_loss(self, fake, real=None, is_disc=False, weight=None):
        if is_disc:
            assert real is not None, 'Discriminator Loss: real is None'
            bce_logit_loss_for_fake = nn.BCEWithLogitsLoss(weight=weight)
            return bce_logit_loss_for_fake(fake, torch.zeros_like(fake).to(self.device)) + self.bce_logit_loss(real, torch.ones_like(real).to(self.device))
        else:
            return self.bce_logit_loss(fake, torch.ones_like(fake).to(self.device))
    
    def LabelSmoothGAN_loss(self, fake, real=None, is_disc=False, weight=None):
        if is_disc:
            assert real is not None, 'Discriminator Loss: real is None'
            bce_logit_loss_for_fake = nn.BCEWithLogitsLoss(weight=weight)
            return bce_logit_loss_for_fake(fake, torch.zeros_like(fake).to(self.device)) + self.bce_logit_loss(real, torch.ones_like(real).to(self.device)*0.9)
        else:
            return self.bce_logit_loss(fake, torch.ones_like(fake).to(self.device)*0.9)

def r1_reg(d_out, x_in):
    # zero-centered gradient penalty for real images
    batch_size = x_in.size(0)
    grad_dout = torch.autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert(grad_dout2.size() == x_in.size())
    reg = 0.5 * grad_dout2.view(batch_size, -1).sum(1).mean(0)
    return reg

class ReconsLoss(nn.Module):
    def __init__(self, type=None, device=None):
        super(ReconsLoss, self).__init__()
        if type is None:
            self.type = 'l1'
        else:
            self.type = type
        losses = {'l1': self.l1_loss, 
                  'l2': self.l2_loss}
        
        self.loss = losses[self.type]
        self.device = device

    def forward(self, fake, real):
        return self.loss(fake, real)
    
    def l1_loss(self, fake, real):
        return torch.mean(torch.abs(fake-real))
    
    def l2_loss(self, fake, real):
        return torch.mean((fake-real)**2)


