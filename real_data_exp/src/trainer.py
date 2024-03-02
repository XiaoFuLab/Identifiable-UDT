import torch
import torch.nn as nn
import os
import numpy as np
from torch import optim
from src.trainer_utils import get_model
import imageio
from torch.optim import lr_scheduler
import wandb
from utils.losses import GANLoss, ReconsLoss
from utils.vgg_loss import VGGLoss
import sys
import copy
from collections import deque
from utils.losses import r1_reg
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from src.trainer_utils import he_init

class Trainer(object):
    def __init__(self, config):
        self.g12 = None
        self.g21 = None
        self.g12_ema = None
        self.g21_ema = None
        self.d1 = None
        self.d2 = None
        self.d1_all = None
        self.d2_all = None
        self.g_optimizer = None
        self.d_optimizer = None
        self.config = config
        
        self.vgg_loss = VGGLoss()

        self.gan_loss_weights = torch.tensor([1.0, 1.0, 1.0, 1.0])

        self.gan_loss = GANLoss(config['gan_criterion'], device='cuda')
        self.recons_criterion = ReconsLoss(config['recons_criterion'])
        self.build_model()

        self.losses = {'loss_gen_total':         deque(maxlen=self.config['console_log_steps']),
                        'loss_gen_adv_1':        deque(maxlen=self.config['console_log_steps']),
                        'loss_gen_adv_2':        deque(maxlen=self.config['console_log_steps']),
                        'loss_gen_cycrecon_x_1': deque(maxlen=self.config['console_log_steps']),
                        'loss_gen_cycrecon_x_2': deque(maxlen=self.config['console_log_steps']),
                        'loss_dis_total':        deque(maxlen=self.config['console_log_steps']),
                        'loss_dis_1':            deque(maxlen=self.config['console_log_steps']),
                        'loss_dis_2':            deque(maxlen=self.config['console_log_steps']),
                        'r1_reg_1':              deque(maxlen=self.config['console_log_steps']), 
                        'r1_reg_2':              deque(maxlen=self.config['console_log_steps']),
                        'loss_vgg_1':            deque(maxlen=self.config['console_log_steps']),
                        'loss_vgg_2':            deque(maxlen=self.config['console_log_steps'])}

    def set_gan_loss_weight(self, num_samples1, num_samples2):
        num_samples1 = torch.tensor(num_samples1).float()
        num_samples2 = torch.tensor(num_samples2).float()
        self.gan_loss_weights = (num_samples1 * num_samples2.sum()) / (num_samples2 * num_samples1.sum())

    def build_model(self):
        """Builds a generator and a discriminator."""
        self.g12, self.g21, self.d1, self.d2, self.d1_all, self.d2_all = get_model(self.config)

        if 'he_init' in self.config.keys() and self.config['he_init']:
            self.g12.apply(he_init)
            self.g21.apply(he_init)
            for d in self.d1:
                d.apply(he_init)
            for d in self.d2:
                d.apply(he_init)
            if self.d1_all is not None:
                self.d1_all.apply(he_init)
            if self.d2_all is not None:
                self.d2_all.apply(he_init)

        self.g12_ema = copy.deepcopy(self.g12)    
        self.g21_ema = copy.deepcopy(self.g21)
        for (param1, param2) in zip(self.g12_ema.parameters(), self.g21_ema.parameters()):
            param1.requires_grad = False   
            param2.requires_grad = False
        self.g12_ema.eval()
        self.g21_ema.eval()

        g_params = list(self.g12.parameters()) + list(self.g21.parameters())
        d_params = []
        for i in range(len(self.d1)):
            d_params += list(self.d1[i].parameters())
            d_params += list(self.d2[i].parameters())
        if self.config['dis_all_w'] > 0:
            d_params += list(self.d1_all.parameters())
            d_params += list(self.d2_all.parameters())
            
        self.g_optimizer = optim.Adam(g_params, self.config['lr'], [self.config['beta1'], self.config['beta2']], weight_decay=self.config['weight_decay'])
        self.d_optimizer = optim.Adam(d_params, self.config['lr'], [self.config['beta1'], self.config['beta2']], weight_decay=self.config['weight_decay'])
        
        def lambda_rule_exponential(num_calls):
            lr_l = 0.95**(max(0, num_calls +1 - (self.config['lr_start_step']/self.config['change_lr_every'])))
            return lr_l
        
        self.schedulers = [lr_scheduler.LambdaLR(self.g_optimizer, lr_lambda=lambda_rule_exponential),
                        lr_scheduler.LambdaLR(self.d_optimizer, lr_lambda=lambda_rule_exponential)]

        if torch.cuda.is_available():
            self.g12.cuda()
            self.g21.cuda()
            self.g12_ema.cuda()
            self.g21_ema.cuda()
            if self.config['dis_all_w'] > 0:
                self.d1_all.cuda()
                self.d2_all.cuda()
            for i in range(len(self.d1)):
                self.d1[i].cuda()
                self.d2[i].cuda()
    
    def merge_images_all(self, sources, targets, recons, k=10):
        labels = ['Source', 'Translation', 'Recons.']
        _, _, h, w = sources.shape
        row = min(int(np.sqrt(self.config['batch_size'])), 8)
        merged = np.zeros([3, row*h, row*w*3])
        for idx, (s, t, r) in enumerate(zip(sources, targets, recons)):
            if idx >= row*row:
                break
            i = idx // row
            j = idx % row
            # print(s.shape, i, j, h, w)
            merged[:, i*h:(i+1)*h, (j*3)*h:(j*3+1)*h] = s
            merged[:, i*h:(i+1)*h, (j*3+1)*h:(j*3+2)*h] = t
            merged[:, i*h:(i+1)*h, (j*3+2)*h:(j*3+3)*h] = r
        return merged.transpose(1,2,0)


    def merge_images_all_plt(self, sources, targets, recons):
        num_images, n_cols = sources.shape[0], 3
        sources, targets, recons = sources*0.5 + 0.5, targets*0.5 + 0.5, recons*0.5 + 0.5
        fig = plt.figure(figsize=(n_cols*10, (num_images)*3))
        grid = ImageGrid(fig, 111, nrows_ncols=(num_images, n_cols), axes_pad=0.1)
        fontsize = 15
        for i in range(num_images):
            grid[i*n_cols].imshow(np.transpose(sources[i],(1, 2, 0)))
            grid[i*n_cols+1].imshow(np.transpose(targets[i],(1, 2, 0)))
            grid[i*n_cols+2].imshow(np.transpose(recons[i],(1, 2, 0)))
            
            # set title for the first row
            if i == 0:
                grid[i*n_cols].set_title('Source', fontsize=fontsize)
                grid[i*n_cols+1].set_title('Translation', fontsize=fontsize)
                grid[i*n_cols+2].set_title('Recons.', fontsize=fontsize)

        # remove the x and y ticks
        for ax in grid:
            ax.set_xticks([])
            ax.set_yticks([])
            
        return fig


    def save_image_eval(self, test_domain1, test_domain2, iterations):
        fake_domain2 = self.g12_ema(test_domain1)
        fake_domain1 = self.g21_ema(test_domain2)
        recons_domain1, recons_domain2 = self.g21_ema(fake_domain2), self.g12_ema(fake_domain1)
        
        domain1, fake_domain1 = test_domain1.detach().cpu().numpy(), fake_domain1.clamp_(-1,1).detach().cpu().numpy()
        domain2 , fake_domain2 = test_domain2.detach().cpu().numpy(), fake_domain2.clamp_(-1,1).detach().cpu().numpy()
        recons_domain1, recons_domain2 = recons_domain1.clamp_(-1,1).detach().cpu().numpy(), recons_domain2.clamp_(-1,1).detach().cpu().numpy()
        
        fig1 = self.merge_images_all(domain1, fake_domain2, recons_domain1)
        path = os.path.join(self.config['sample_path'], f'AB_{iterations}.png')
        # fig1.savefig(path, format='png', dpi=300, bbox_inches='tight')
        
        fig2 = self.merge_images_all(domain2, fake_domain1, recons_domain2)
        path = os.path.join(self.config['sample_path'], f'BA_{iterations}.png')
        # fig1.savefig(path, format='png', dpi=300, bbox_inches='tight')

        return fig1, fig2

    def step(self, x_1, l_1, x_2, l_2):
        
        # Update Discriminator
        self.d_optimizer.zero_grad()
        
        # translate images
        x_12 = self.g12(x_1)
        x_21 = self.g21(x_2)
        
        self.loss_dis_1, self.loss_dis_2 = 0.0, 0.0
        self.r1_reg_1, self.r1_reg_2 = torch.tensor([0.0]).cuda(), torch.tensor([0.0]).cuda()
        # D loss
        if self.config['r1_reg_w'] > 0:
            x_1.requires_grad_()
            x_2.requires_grad_()

        out1, _ = self.d1[0](x_1, l_1)
        out2, _ = self.d2[0](x_2, l_2)
        out_fake1, cond_id1 = self.d1[0](x_21.detach(), l_2)
        weight1 = 1 / torch.tensor([self.gan_loss_weights[i] for i in cond_id1]).to(l_1.device)
        out_fake2, cond_id2 = self.d2[0](x_12.detach(), l_1)
        weight2 = torch.tensor([self.gan_loss_weights[i] for i in cond_id2]).to(l_1.device)

        self.loss_dis_1 += self.gan_loss(out_fake1, real=out1, is_disc=True, weight=weight1)
        self.loss_dis_2 += self.gan_loss(out_fake2, real=out2, is_disc=True, weight=weight2)
        if self.config['r1_reg_w'] > 0:    
            self.r1_reg_1 += r1_reg(out1, x_1)
            self.r1_reg_2 += r1_reg(out2, x_2)

        self.loss_dis_total = self.config['dis_w'] * self.loss_dis_1 + \
                                self.config['dis_w'] * self.loss_dis_2 + \
                                self.config['r1_reg_w'] * self.r1_reg_1 + \
                                self.config['r1_reg_w'] * self.r1_reg_2
        
        self.loss_dis_total.backward()
        self.d_optimizer.step()
        
        # update Generator
        self.g_optimizer.zero_grad()

        # back translate images
        x_121 = self.g21(x_12)
        x_212 = self.g12(x_21)
        
        # reconstruction loss
        self.loss_gen_cycrecon_x_1 = self.recons_criterion(x_121, x_1) 
        self.loss_gen_cycrecon_x_2 = self.recons_criterion(x_212, x_2)

        self.loss_vgg_1 = self.vgg_loss(x_121, x_1)
        self.loss_vgg_2 = self.vgg_loss(x_212, x_2)
        
        # GAN loss
        self.loss_gen_adv_1, self.loss_gen_adv_2 = 0.0, 0.0

        out_fake1, _ = self.d1[0](x_21, l_2)
        out_fake2, _ = self.d2[0](x_12, l_1)
        self.loss_gen_adv_1 += self.gan_loss(out_fake1, is_disc=False)
        self.loss_gen_adv_2 += self.gan_loss(out_fake2, is_disc=False)      
        
        self.loss_gen_total = self.config['gen_w'] * self.loss_gen_adv_1 + \
                            self.config['gen_w'] * self.loss_gen_adv_2 + \
                            self.config['recons_w'] * self.loss_gen_cycrecon_x_1 + \
                            self.config['recons_w'] * self.loss_gen_cycrecon_x_2 + \
                            self.config['vgg_w'] * self.loss_vgg_1 + \
                            self.config['vgg_w'] * self.loss_vgg_2
        self.loss_gen_total.backward()
        self.g_optimizer.step()

        if self.config['use_ema']:
            self.moving_average(self.g12, self.g12_ema, beta=0.999)
            self.moving_average(self.g21, self.g21_ema, beta=0.999)
        else:
            self.moving_average(self.g12, self.g12_ema, beta=0.0)
            self.moving_average(self.g21, self.g21_ema, beta=0.0)

        # Update average Losses
        self.losses['loss_gen_total'].append(self.loss_gen_total.item())
        self.losses['loss_gen_adv_1'].append(self.loss_gen_adv_1.item())
        self.losses['loss_gen_adv_2'].append(self.loss_gen_adv_2.item())
        self.losses['loss_gen_cycrecon_x_1'].append(self.loss_gen_cycrecon_x_1.item())
        self.losses['loss_gen_cycrecon_x_2'].append(self.loss_gen_cycrecon_x_2.item())
        self.losses['loss_vgg_1'].append(self.loss_vgg_1.item())
        self.losses['loss_vgg_2'].append(self.loss_vgg_2.item())
        self.losses['loss_dis_total'].append(self.loss_dis_total.item())
        self.losses['loss_dis_1'].append(self.loss_dis_1.item())
        self.losses['loss_dis_2'].append(self.loss_dis_2.item())
        self.losses['r1_reg_1'].append(self.r1_reg_1.item())
        self.losses['r1_reg_2'].append(self.r1_reg_2.item())
        
    
    @staticmethod
    def moving_average(model, model_test, beta=0.999):
        for param, param_test in zip(model.parameters(), model_test.parameters()):
            param_test.data = torch.lerp(param.data, param_test.data, beta)

    def sample(self, x_1, x_2):
        x_121, x_212, x_21, x_12 = [], [], [], []
        for i in range(x_1.size(0)):
            x_21.append(self.g21_ema(x_2[i].unsqueeze(0)))
            x_12.append(self.g12_ema(x_1[i].unsqueeze(0)))
            x_121.append(self.g21_ema(x_12[-1]))
            x_212.append(self.g12_ema(x_21[-1]))
        x_121, x_212 = torch.cat(x_121), torch.cat(x_212)
        x_12, x_21 = torch.cat(x_12), torch.cat(x_21)
        
        return x_1.detach().cpu(), x_121.detach().cpu(), x_12.detach().cpu(), x_2.detach().cpu(), x_212.detach().cpu(), x_21.detach().cpu()

    def update_learning_rate(self, iterations):
        if iterations%self.config['change_lr_every'] == 0: 
            if self.schedulers is not None:
                for scheduler in self.schedulers:
                    scheduler.step()
            
    def save_checkpoint(self, filename, iterations):
        params = {
            'g12': self.g12.state_dict(),
            'g21': self.g21.state_dict(),
            'g12_ema': self.g12_ema.state_dict(),
            'g21_ema': self.g21_ema.state_dict(),
            'd1': [net.state_dict() for net in self.d1],
            'd2': [net.state_dict() for net in self.d2],
            'g_optimizer': self.g_optimizer.state_dict(),
            'd_optimizer': self.d_optimizer.state_dict(),
            'step': iterations
        }
        torch.save(params, filename)
        
    def load_checkpoint(self, checkpoint_path):
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            self.g12.load_state_dict(checkpoint['g12'])
            self.g21.load_state_dict(checkpoint['g21'])
            self.g12_ema.load_state_dict(checkpoint['g12_ema'])
            self.g21_ema.load_state_dict(checkpoint['g21_ema'])
            for i in range(len(self.d1)):
                self.d1[i].load_state_dict(checkpoint['d1'][i])
                self.d2[i].load_state_dict(checkpoint['d2'][i])
            self.g_optimizer.load_state_dict(checkpoint['g_optimizer'])
            self.d_optimizer.load_state_dict(checkpoint['d_optimizer'])
            return checkpoint['step']
        else:
            print(f'Loading Checkpoint Failed. Checkpoint {checkpoint_path} does not exist.')
                     
    def log_err_wandb(self, iterations):
        if self.config['use_wandb']:
            result = {}
            for key in self.losses:
                result[key] = np.mean(self.losses[key])
            wandb.log(result, step=iterations)
        
    def log_err_console(self, it):
        print('iter: %d, loss_gen_total: %.4f, loss_gen_adv_1: %.4f, loss_gen_adv_2: %.4f, loss_gen_cycrecon_x_1: %.4f, loss_gen_cycrecon_x_2: %.4f, loss_dis_total: %.4f, loss_dis_1: %.4f, loss_dis_2: %.4f' %  \
                    (it, self.loss_gen_total.item(), self.loss_gen_adv_1.item(), self.loss_gen_adv_2.item(), self.loss_gen_cycrecon_x_1.item(), self.loss_gen_cycrecon_x_2.item(), self.loss_dis_total.item(), self.loss_dis_1.item(), self.loss_dis_2.item()))
        