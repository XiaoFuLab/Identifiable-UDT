
from model import Translator as Stargan_Translator
from model import Discriminator as MultiTaskDiscriminator
import sys
import torch.nn as nn

def he_init(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)

def get_model(config):
    """Builds a generator and a discriminator."""

    num_downsample = config['gen']['num_downsample'] if 'num_downsample' in config['gen'] else None
    use_adain = config['gen']['use_adain'] if 'use_adain' in config['gen'] else False
    w_hpf = config['gen']['w_hpf'] if 'w_hpf' in config['gen'] else use_adain

    g12 = Stargan_Translator(img_size=config['new_size'], w_hpf=w_hpf, num_downsample=num_downsample, use_adain=use_adain)
    g21 = Stargan_Translator(img_size=config['new_size'], w_hpf=w_hpf, num_downsample=num_downsample, use_adain=use_adain)
   
    d1 = [MultiTaskDiscriminator(img_size=config['new_size'], num_domains=config['num_conditionals']), ]
    d2 = [MultiTaskDiscriminator(img_size=config['new_size'], num_domains=config['num_conditionals']), ]

    return g12, g21, d1, d2, None, None