import torch
import torch.nn as nn
import os
import numpy as np

from torch.autograd import Variable
from torch import optim
import imageio
import datetime
from torch.optim import lr_scheduler
import wandb
from model import Generator, Critic
import matplotlib.pyplot as plt

class Trainer(object):
    def __init__(self, config, train_loader, device, num_conditionals=1):
        self.train_loader = train_loader
        self.g12, self.g21 = None, None
        self.d1, self.d2 = [], []
        self.g_optimizer = None
        self.d_optimizer = None
        self.config = config
        self.num_conditionals = num_conditionals
        self.device = device
        self.trans_err = 0
        
        if not os.path.isdir(self.config['model_path']):
            os.makedirs(self.config['model_path'])
        if not os.path.isdir(self.config['result_path']):
            os.makedirs(self.config['result_path'])
    
        self.build_model()
        self.model_path = self.config['model_path']

            
    def set_model_path(self, path):
        if not os.path.isdir(path):
            os.makedirs(path)
        self.model_path = path
        
    def build_model(self):
        """Builds a generator and a discriminator."""
        
        self.g12 = Generator(input_dim=self.config['data_dim'], 
                             output_dim=self.config['data_dim'], 
                             hidden_units=self.config['gen']['hidden_dim'], 
                             use_batchnorm=self.config['gen']['use_batchnorm'], 
                             num_layers=self.config['gen']['num_layers'], 
                             activation_func=self.config['gen']['activation_func'])
        
        self.g21 = Generator(input_dim=self.config['data_dim'],
                            output_dim=self.config['data_dim'],
                            hidden_units=self.config['gen']['hidden_dim'],
                            use_batchnorm=self.config['gen']['use_batchnorm'],
                            num_layers=self.config['gen']['num_layers'],
                            activation_func=self.config['gen']['activation_func'])

        for i in range(self.num_conditionals):
            self.d1.append(Critic(input_dim=self.config['data_dim'], hidden_units=self.config['critic']['hidden_dim']))
            self.d2.append(Critic(input_dim=self.config['data_dim'], hidden_units=self.config['critic']['hidden_dim']))
        
        g_params = list(self.g12.parameters()) + list(self.g21.parameters())
        d_params = []
        for i in range(self.num_conditionals):
            d_params += list(self.d1[i].parameters())
            d_params += list(self.d2[i].parameters())
            
        self.g_optimizer = optim.Adam(g_params, self.config['lr_gen'], [self.config['beta1'], self.config['beta2']], weight_decay=self.config['weight_decay'])
        self.d_optimizer = optim.Adam(d_params, self.config['lr_dis'], [self.config['beta1'], self.config['beta2']], weight_decay=self.config['weight_decay'])

        if torch.cuda.is_available():
            self.g12.cuda()
            self.g21.cuda()
            for i in range(self.num_conditionals):
                self.d1[i].to(self.device)
                self.d2[i].to(self.device)
   
    def recon_criterion(self, input, target):
        return torch.mean((input - target)**2)
    
    def reset_grad(self):
        """Zeros the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def gen_update(self, x_1, l_1, x_2, l_2):
        self.g_optimizer.zero_grad()

        # translate images
        x_12 = self.g12(x_1)
        x_21 = self.g21(x_2)

        # back translate images
        x_121 = self.g21(x_12)
        x_212 = self.g12(x_21)
        
        # reconstruction loss
        self.loss_gen_cycrecon_x_1 = self.recon_criterion(x_121, x_1) 
        self.loss_gen_cycrecon_x_2 = self.recon_criterion(x_212, x_2)
        
        # GAN loss
        self.loss_gen_adv_1, self.loss_gen_adv_2 = 0.0, 0.0
        for i in range(self.num_conditionals):
            x_12i = x_12[l_1[range(len(l_1)), i]==1]
            x_21i = x_21[l_2[range(len(l_2)), i]==1]
            if len(x_12i)>0:
                out_fake = self.d2[i](x_12i)
                for out_fake_ in out_fake:
                    self.loss_gen_adv_1 += torch.mean((out_fake_-1)**2)
            
            if len(x_21i)>0:
                out_fake = self.d1[i](x_21i)
                for out_fake_ in out_fake:
                    self.loss_gen_adv_2 += torch.mean((out_fake_-1)**2)
        
        # total loss
        self.loss_gen_total = self.config['gen_w'] * self.loss_gen_adv_1 / self.num_conditionals + \
                              self.config['gen_w'] * self.loss_gen_adv_2 / self.num_conditionals + \
                              self.config['recons_w'] * self.loss_gen_cycrecon_x_1 + \
                              self.config['recons_w'] * self.loss_gen_cycrecon_x_2
        self.loss_gen_total.backward()
        self.g_optimizer.step()
        
    def dis_update(self, x_1, l_1, x_2, l_2):
        self.d_optimizer.zero_grad()
        
        # translate images
        x_12 = self.g12(x_1)
        x_21 = self.g21(x_2)
        
        self.loss_dis_1, self.loss_dis_2 = 0.0, 0.0
        # D loss
        for i in range(self.num_conditionals):
            x_1i = x_1[l_1[range(len(l_1)), i]==1]
            x_12i = x_12[l_1[range(len(l_1)), i]==1]
            
            x_2i = x_2[l_2[range(len(l_2)), i]==1]
            x_21i = x_21[l_2[range(len(l_2)), i]==1]
            
            if len(x_1i)>0:
                out = self.d1[i](x_1i)
                out_fake = self.d2[i](x_12i.detach())
                for (out_, out_fake_) in zip(out, out_fake):
                    self.loss_dis_1 += torch.mean((out_-1)**2)
                    self.loss_dis_2 += torch.mean(out_fake_**2)

            if len(x_2i)>0:
                out = self.d2[i](x_2i)
                out_fake = self.d1[i](x_21i.detach())
                for (out_, out_fake_) in zip(out, out_fake):
                    self.loss_dis_1 += torch.mean(out_fake_**2)
                    self.loss_dis_2 += torch.mean((out_-1)**2)
            
        self.loss_dis_total = self.config['dis_w'] * self.loss_dis_1 + self.config['dis_w'] * self.loss_dis_2
        self.loss_dis_total.backward()
        self.d_optimizer.step()
    
    def sample(self, x_1, x_2):
        self.g12.eval()
        self.g21.eval()
        x_121, x_212, x_21, x_12 = [], [], [], []
        for i in range(x_1.size(0)):
            x_21.append(self.g21(x_2[i].unsqueeze(0)))
            x_12.append(self.g12(x_1[i].unsqueeze(0)))
            x_121.append(self.g21(x_12[-1]))
            x_212.append(self.g12(x_21[-1]))
        x_121, x_212 = torch.cat(x_121), torch.cat(x_212)
        x_12, x_21 = torch.cat(x_12), torch.cat(x_21)

        self.g12.train()
        self.g21.train()
        
        return x_1.detach().cpu(), x_121.detach().cpu(), x_12.detach().cpu(), x_2.detach().cpu(), x_212.detach().cpu(), x_21.detach().cpu()
        

    def update_learning_rate(self):
        if self.dis_scheduler is not None:
            self.dis_scheduler.step()
        if self.gen_scheduler is not None:
            self.gen_scheduler.step()
            
    def load_model(self, iters):
        g12_path = os.path.join(self.model_path, 'g12-%d.pkl' %(iters))
        g21_path = os.path.join(self.model_path, 'g21-%d.pkl' %(iters))
        d1_path = os.path.join(self.model_path , 'd1-%d.pkl' %(iters))
        d2_path = os.path.join(self.model_path , 'd2-%d.pkl' %(iters))
        self.g12.load_state_dict(torch.load(g12_path))
        self.g21.load_state_dict(torch.load(g21_path))
        for i in range(self.num_conditionals):
            self.d1[i].load_state_dict(torch.load(d1_path)[i])
            self.d2[i].load_state_dict(torch.load(d2_path)[i])
    
    def save_model(self, iters):
        g12_path = os.path.join(self.model_path, 'g12-%d.pkl' %(iters+1))
        g21_path = os.path.join(self.model_path, 'g21-%d.pkl' %(iters+1))
        d1_path = os.path.join(self.model_path , 'd1-%d.pkl' %(iters+1))
        d2_path = os.path.join(self.model_path , 'd2-%d.pkl' %(iters+1))
        torch.save(self.g12.state_dict(), g12_path)
        torch.save(self.g21.state_dict(), g21_path)
        torch.save([net.state_dict() for net in self.d1], d1_path)
        torch.save([net.state_dict() for net in self.d1], d2_path)
    
    
    def translation_err(self, x_1, x_2, x_12, x_21):
        # aligned data points in the two domains
        mse_loss = torch.nn.MSELoss()
        self.trans_err = (mse_loss(x_1, x_21) + mse_loss(x_2, x_12))/2.0
        if self.config['use_wandb']:
            wandb.log({'translation error': self.trans_err.item()})
        return self.trans_err.item()
    
    def log_err_wandb(self):
        wandb.log({'loss_gen_total': self.loss_gen_total.item(),
                    'loss_gen_adv_1': self.loss_gen_adv_1.item(),
                    'loss_gen_adv_2': self.loss_gen_adv_2.item(),
                    'loss_gen_cycrecon_x_1': self.loss_gen_cycrecon_x_1.item(),
                    'loss_gen_cycrecon_x_2': self.loss_gen_cycrecon_x_2.item(),
                    'loss_dis_total': self.loss_dis_total.item(),
                    'loss_dis_1': self.loss_dis_1.item(),
                    'loss_dis_2': self.loss_dis_2.item()})
        
    def log_err_console(self, it):
        print('iter: %d, loss_gen_total: %.4f, loss_gen_adv_1: %.4f, loss_gen_adv_2: %.4f, loss_gen_cycrecon_x_1: %.4f, loss_gen_cycrecon_x_2: %.4f, loss_dis_total: %.4f, loss_dis_1: %.4f, loss_dis_2: %.4f' %  \
                    (it, self.loss_gen_total.item(), self.loss_gen_adv_1.item(), self.loss_gen_adv_2.item(), self.loss_gen_cycrecon_x_1.item(), self.loss_gen_cycrecon_x_2.item(), self.loss_dis_total.item(), self.loss_dis_1.item(), self.loss_dis_2.item()))
        
    def plot_scatter(self, x_1, x_2, x_12, x_21, save_filename=None):
        # The samples should be aligned in the two domains
        # scatter plot x_1 and x_21 side by side. Similarly for x_2 and x_12 using 2x2 subplots
        # color the data points according to radius and angle of the 2 dimensional data points
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        axs[0, 0].scatter(x_1[:, 0], x_1[:, 1], c=np.arctan2(x_1[:, 1], x_1[:, 0]), cmap='hsv')
        axs[0, 0].set_title('x_1')
        axs[0, 1].scatter(x_21[:, 0], x_21[:, 1], c=np.arctan2(x_1[:, 1], x_1[:, 0]), cmap='hsv')
        axs[0, 1].set_title('x_21')
        
        axs[1, 0].scatter(x_2[:, 0], x_2[:, 1], c=np.arctan2(x_1[:, 1], x_1[:, 0]), cmap='hsv')
        axs[1, 0].set_title('x_2')
        axs[1, 1].scatter(x_12[:, 0], x_12[:, 1], c=np.arctan2(x_1[:, 1], x_1[:, 0]), cmap='hsv')
        axs[1, 1].set_title('x_12')
        
        if self.config['use_wandb']:
            wandb.log({"plot": wandb.Image(fig)})
        if save_filename:
            fig.savefig(save_filename)
        
