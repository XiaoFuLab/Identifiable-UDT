
import os
from src.trainer import Trainer
from utils.data_loader import get_loader
import torch
import argparse
import yaml
import sys
from tqdm import tqdm
import imageio
import numpy as np

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/celeba2bitmoji_cond.yaml', help='Path to the config file.')
parser.add_argument('--dest_dir', type=str, default='results', help='Path to the config file.')
parser.add_argument('--checkpoint_path', type=str, default='', help='Path to the checkpoint.')
opts = parser.parse_args()


print('Testing on {}'.format(opts.config))
# Load experiment setting
with open(opts.config, 'r') as stream:
    config = yaml.safe_load(stream)

config['model_path'] = os.path.join(config['model_path'], config['run_name'])

if not os.path.exists(opts.dest_dir):
    os.makedirs(opts.dest_dir)

print('Preparing dataset...')
config['batch_size'] = 8
test_loader1 = get_loader(config, domain=config['domain1'], train=False)
test_loader2 = get_loader(config, domain=config['domain2'], train=False)


trainer = Trainer(config)

if opts.checkpoint_path != '':
    print('Loading checkpoint {}'.format(opts.checkpoint_path))
    iterations = trainer.load_checkpoint(opts.checkpoint_path)

elif os.path.isfile(os.path.join(config['model_path'], 'checkpoint-current.pt')):            
    print('Loading latest checkpoint {}'.format(os.path.join(config['model_path'], 'checkpoint-current.pt')))
    iterations = trainer.load_checkpoint(os.path.join(config['model_path'], 'checkpoint-current.pt'))
else:
    print('No checkpoint found at {}'.format(os.path.join(config['model_path'], 'checkpoint-current.pt')))
    sys.exit()


def merge_images_all(sources, targets, recons, k=10):
    _, _, h, w = sources.shape
    row = min(int(np.sqrt(config['batch_size'])), 8)
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

MAX_SAMPLES = 100

with torch.no_grad():
    save_dict = {'real_A': [], 'real_B': [], 'fake_A': [], 'fake_B': [], 'rec_A': [], 'rec_B': []}
    for i, ((real_A,_), (real_B,_)) in tqdm(enumerate(zip(test_loader1, test_loader2))):
        
        real_A = real_A.to('cuda')
        real_B = real_B.to('cuda')
        fake_A = trainer.g21_ema(real_B)
        fake_B = trainer.g12_ema(real_A)
        rec_A =  trainer.g21_ema(fake_B)
        rec_B =  trainer.g12_ema(fake_A)
        
        save_dict['real_A'].append((real_A+1.0)/2.0)
        save_dict['real_B'].append((real_B+1.0)/2.0)
        save_dict['fake_A'].append((fake_A+1.0)/2.0)
        save_dict['fake_B'].append((fake_B+1.0)/2.0)
        save_dict['rec_A'].append((rec_A+1.0)/2.0)
        save_dict['rec_B'].append((rec_B+1.0)/2.0)
        
        if i*config['batch_size'] > MAX_SAMPLES:
            break

    # save_dict['real_A'] = torch.cat(save_dict['real_A'], dim=0)
    # save_dict['real_B'] = torch.cat(save_dict['real_B'], dim=0)
    # save_dict['fake_A'] = torch.cat(save_dict['fake_A'], dim=0)
    # save_dict['fake_B'] = torch.cat(save_dict['fake_B'], dim=0)
    # save_dict['rec_A'] = torch.cat(save_dict['rec_A'], dim=0)
    # save_dict['rec_B'] = torch.cat(save_dict['rec_B'], dim=0)

    # # crop 32x32 mnist images to 28x28
    # if 'mnist' in opts.config and save_dict['real_A'].shape[-1] == 32:
    #     save_dict = {k: v[:, :, 2:-2, 2:-2] for k, v in save_dict.items()}
    
    # save image as png in result_AB and result_BA. Specifically, merge (real_B, fake_A, and rec_B) and save in result_BA and (real_A, fake_B, rec_A) in result_AB
    # convert all images to PIL images and save them

    if not os.path.exists(os.path.join(opts.dest_dir, 'result_AB')):
        os.makedirs(os.path.join(opts.dest_dir, 'result_AB'))
    if not os.path.exists(os.path.join(opts.dest_dir, 'result_BA')):
        os.makedirs(os.path.join(opts.dest_dir, 'result_BA'))

    for i in range(len(save_dict['real_A'])):
        print('saving image AB: ', i)
        domain1 = save_dict['real_A'][i].cpu().numpy()
        fake_domain2 = save_dict['fake_B'][i].cpu().numpy()
        recons_domain1 = save_dict['rec_A'][i].cpu().numpy()
        merged1 = merge_images_all(domain1, fake_domain2, recons_domain1)
        path = os.path.join(opts.dest_dir, 'result_AB', f'AB_{i}.png')
        imageio.imwrite(path, merged1)

    for i in range(len(save_dict['real_B'])):
        print('saving image BA: ', i)
        domain1 = save_dict['real_B'][i].cpu().numpy()
        fake_domain2 = save_dict['fake_A'][i].cpu().numpy()
        recons_domain1 = save_dict['rec_B'][i].cpu().numpy()
        merged1 = merge_images_all(domain1, fake_domain2, recons_domain1)
        path = os.path.join(opts.dest_dir, 'result_BA', f'BA_{i}.png')
        imageio.imwrite(path, merged1)

                
    

    
