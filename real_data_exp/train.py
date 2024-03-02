
import os
from src.trainer import Trainer
from utils.data_loader import get_loader
import torch
import argparse
import yaml
import sys
import wandb
import datetime
import numpy as np

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/celeba2bitmoji_cond.yaml', help='Path to the config file.')
parser.add_argument('--resume', action='store_true')
parser.add_argument('--debug', action='store_true')
opts = parser.parse_args()


# Load experiment setting
with open(opts.config, 'r') as stream:
    config = yaml.safe_load(stream)

print(config)

# set hpc path if running on hpc
HPC_PATH='/nfs/hpc/share/shressag/Projects/Data-Split-Domain-Translation'
if os.path.isdir(HPC_PATH):
    os.environ['WANDB_DIR'] = os.path.join(HPC_PATH, 'wandb')
    config['model_path'] = os.path.join(HPC_PATH, config['model_path'], config['run_name'])
    config['sample_path'] = os.path.join(HPC_PATH, config['sample_path'], config['run_name'])
    config['data_path'] = os.path.join(HPC_PATH, config['data_path'])
else:
    config['model_path'] = os.path.join(config['model_path'], config['run_name'])
    config['sample_path'] = os.path.join(config['sample_path'], config['run_name'])

# create directories if not exist                                                           
if not os.path.exists(config['model_path']):
    os.makedirs(config['model_path'])
if not os.path.exists(config['sample_path']):
    os.makedirs(config['sample_path'])

print('Preparing dataset...')
train_loader1 = get_loader(config, domain=config['domain1'], train=True)
train_loader2 = get_loader(config, domain=config['domain2'], train=True)
test_loader1 = get_loader(config, domain=config['domain1'], train=False)
test_loader2 = get_loader(config, domain=config['domain2'], train=False)

if not opts.debug:
    test_display_images1 = torch.stack([test_loader1.dataset[i][0] for i in range(min(16, len(test_loader1.dataset)))]).cuda()
    test_display_images2 = torch.stack([test_loader2.dataset[i][0] for i in range(min(16, len(test_loader2.dataset)))]).cuda()



trainer = Trainer(config)
if config['adjust_class_imbalance']:
    cond_size1 = train_loader1.dataset.get_conditional_sizes()
    cond_size2 = train_loader2.dataset.get_conditional_sizes()
    trainer.set_gan_loss_weight(cond_size1, cond_size2)

run_name = config['run_name'] if config['run_name'] != '' else datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
if config['use_wandb'] and not opts.debug:
    wandb.init(
        # set the wandb project where this run will be logged
        project='domain_translation',
        group=config['group_name'] if 'group_name' in config else None,
        name=run_name + f'-{datetime.datetime.now().strftime("%Y-%m-%d")}-{datetime.datetime.now().strftime("%H:%M:%S")}',
        resume=opts.resume,
        # track hyperparameters and run metadata
        config=config
    )
        
iterations = 1

if opts.resume:
    if os.path.isfile(os.path.join(config['model_path'], 'checkpoint-current.pt')):            
        print('Loading latest checkpoint {}'.format(os.path.join(config['model_path'], 'checkpoint-current.pt')))
        iterations = trainer.load_checkpoint(os.path.join(config['model_path'], 'checkpoint-current.pt'))
    else:
        print('No checkpoint found at {}'.format(os.path.join(config['model_path'], 'checkpoint-current.pt')))
        sys.exit()
            
while iterations <= config['train_iters']:
    for (images_1, labels_1), (images_2, labels_2) in zip(train_loader1, train_loader2):
        if config['lr_decay']:
            trainer.update_learning_rate(iterations)
        images_1, images_2= images_1.cuda().detach(), images_2.cuda().detach()
        labels_1, labels_2 = labels_1.cuda(), labels_2.cuda()

        if config['num_conditionals'] == 1:
            labels_1 = torch.ones(labels_1.shape[0], 1).cuda()
            labels_2 = torch.ones(labels_2.shape[0], 1).cuda()
        
        trainer.step(images_1, labels_1, images_2, labels_2)
        torch.cuda.synchronize()

        # print the log info
        if iterations % config['console_log_steps'] == 0:
            trainer.log_err_console(iterations)
            if not opts.debug:
                trainer.log_err_wandb(iterations)
                
        # Test model and save the sampled images
        if iterations % config['test_sample_steps'] == 0:
            merged1, merged2 = trainer.save_image_eval(test_display_images1, test_display_images2, iterations)

            if config['use_wandb'] and not opts.debug:
                wandb.log({'A2B': wandb.Image(merged1), 'B2A': wandb.Image(merged2)}, step=iterations)
                # wandb_images = wandb.Image(np.concatenate([merged1, merged2], axis=0), caption="Top: Output, Bottom: Input")
                # wandb.log({'example': wandb_images})
            
        # Save model checkpoints
        if iterations % config['save_checkpoint_steps'] == 0:
            trainer.save_checkpoint(os.path.join(config['model_path'], 'checkpoint-%d.pt' %(iterations)), iterations)
            trainer.save_checkpoint(os.path.join(config['model_path'], 'checkpoint-current.pt'), iterations)
            print('Saved model checkpoint')

        iterations += 1