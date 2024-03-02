import torch
from torchvision import transforms
from typing import Union
from typing_extensions import Literal
from torch.utils.data import Dataset
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from utils.data import ImageFolder


class LabeledDataset(Dataset):
    def __init__(self, 
                 input_folder='../../../datasets_i2i/celeba2bitmoji', 
                 domain='A', 
                 keys=None,
                 train=True, 
                 rotate=False, 
                 crop=True, 
                 new_size=128,
                 horizontal_flip=False,
                 discriminator_dataset=False,
                 include_negatives=False,
                 low_pass=False):
        
        assert os.path.exists(input_folder), 'input_folder {} does not exist'.format(input_folder)
        self.zero_pad = False
        if 'MNIST' in input_folder and new_size==32:
            self.zero_pad = True

        if keys is not None:
            self.keys = keys
        elif 'celebahq2bitmoji' in input_folder:   
            self.keys = ['Male', '~Male', 'Black_Hair', '~Black_Hair']  
        elif 'edges2shoes' in input_folder:
            self.keys=['Shoes','Sandals','Slippers','Boots']
        else:
            self.keys=['dummy']

        if domain=='A':
            folder_path = os.path.join(input_folder, 'trainA' if train else 'testA')
            attribute_path = os.path.join(input_folder, 'trainA_attr.csv' if train else 'testA_attr.csv')
        else:
            folder_path = os.path.join(input_folder, 'trainB' if train else 'testB')
            attribute_path = os.path.join(input_folder, 'trainB_attr.csv' if train else 'testB_attr.csv')
        
        if self.zero_pad:
            transform_list = [
                transforms.Pad([2,], fill=0, padding_mode='constant'),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5),
                                    (0.5, 0.5, 0.5))]
        else:
            transform_list = [transforms.Resize((new_size, new_size), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5),
                                    (0.5, 0.5, 0.5))]

        if low_pass:
            transform_list = [transforms.Resize((new_size//2, new_size//2), interpolation=transforms.InterpolationMode.BICUBIC),] + transform_list

            

        rand_rotate = [transforms.RandomRotation((-90, -90), fill=255)] if rotate else []
        horizontal_flip = [transforms.RandomHorizontalFlip(p=0.5)] if (train and horizontal_flip) else []
        transform_list = horizontal_flip + rand_rotate + transform_list
            
        self.transform = transforms.Compose(transform_list)

        image_data = self.read_attr_file(attribute_path, folder_path)
        self.files = image_data['image_id'].values
        # print the columns of the dataframe image_data
        # print('###############\nall attributes', image_data.columns)
        self.labels = image_data[self.keys].copy()
        for key in self.keys:
            self.labels[key] = list(map(lambda x: max(int(x), 0), self.labels[key]))
            if include_negatives:
                self.labels['~'+key] = list(map(lambda x: 0 if x else 1, self.labels[key]))
        self.image_by_labels = []
        for key in self.keys:
            self.image_by_labels.append([self.files[j] for j in range(len(self.files)) if self.labels[key][j]==1])
            if include_negatives:
                self.image_by_labels.append([self.files[j] for j in range(len(self.files)) if self.labels['~'+key][j]==1])
        self.discriminator_dataset = discriminator_dataset
        
    def get_conditional_sizes(self):
        return [len(self.image_by_labels[i]) for i in range(len(self.image_by_labels))]

    def read_attr_file(self, attr_path, image_dir):
        if os.path.exists(attr_path):
            f = open(attr_path)
            lines = f.readlines()
            lines = list(map(lambda line: line.strip(), lines))
            columns = lines[0].split(',')
            lines = lines[1:]

            items = map(lambda line: line.split(','), lines)

        else:
            print("Attribute file not found. Creating a dummy attribute file")
            # create a dataframe with dummy column and 1s as items
            columns = ['image_id', 'dummy']
            items = map(lambda x: [x, 1], os.listdir(image_dir))

        df = pd.DataFrame( items, columns=columns )
        df['image_id'] = df['image_id'].map( lambda x: os.path.join( image_dir, x ) )
        return df

    def __len__(self):
        if not self.discriminator_dataset:
            return len(self.files)
        else:
            return min([len(self.image_by_labels[i]) for i in range(len(self.keys))])
    
    def __getitem__(self, idx):
        if not self.discriminator_dataset:
            image = plt.imread(self.files[idx]).copy()
            if len(image.shape) == 2:
                image = np.repeat(np.expand_dims(image, axis=2), 3, axis=2)
            

            if image.dtype == 'float32' and image.max()<=1.0 and image.min()>=0.0:
                image = (image*255).astype(np.uint8)
            image = self.transform(Image.fromarray(image))
            # image = self.transform(image)
            label = torch.tensor(self.labels.iloc[idx])
            return image, label
        else:
            images = []
            for i in range(len(self.keys)):
                # ignore idx and randomly sample an index from the length of the list
                randidx = np.random.randint(0, len(self.image_by_labels[i]))
                image = plt.imread(self.image_by_labels[i][randidx]).copy()
                if len(image.shape) == 2:
                    image = np.repeat(np.expand_dims(image, axis=2), 3, axis=2)

                if image.dtype == 'float32' and image.max()<=1.0 and image.min()>=0.0:
                    image = (image*255).astype(np.uint8)
                image = self.transform(Image.fromarray(image))
                images.append(image)
            return images


class DefaultDataset(Dataset):
    def __init__(self, input_folder='../../dataset_i2i/selfie2anmie', domain='A', train=True, rotate=False, crop=True, new_size=128, max_rotation=0, horizontal_flip=False):
        if domain=='A':
            folder_path = os.path.join(input_folder, 'trainA' if train else 'testA')
        else:
            folder_path = os.path.join(input_folder, 'trainB' if train else 'testB')

        transform_list = [transforms.Resize((new_size, new_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                                (0.5, 0.5, 0.5))]
        rand_rotate = [transforms.RandomRotation((-90-max_rotation, -90+max_rotation), fill=255)] if rotate else []
        horizontal_flip = [transforms.RandomHorizontalFlip(p=0.5)] if (train and horizontal_flip) else []
        transform_list = horizontal_flip + rand_rotate + transform_list
        
        transform = transforms.Compose(transform_list)
        
        self.image_set = ImageFolder(folder_path, transform=transform)
        self.labels = torch.ones(len(self.image_set))
            
    def __len__(self):
        return len(self.image_set)
    
    def __getitem__(self, index):
        return self.image_set[index], self.labels[index]
    
    
def get_loader(config, 
               domain:Union[str, Literal['mnist', 'rotatedmnist', 'shoes_edges', 'shoes', 'rotatedshoes', 'bitmoji', 'celebaForBitmoji', 'shoes_edges_2_cond', 'rotatedshoes_2_cond' ]],
               train=True,
               discriminator_dataset=False) -> torch.utils.data.DataLoader:
    """Builds and returns Dataloader selected dataset."""
    if domain == 'mnist':
        dataset = LabeledDataset(input_folder=config['data_path'], 
                                 keys=['0','1', '2', '3', '4', '5', '6', '7', '8', '9'], 
                                 domain='A', 
                                 rotate=False, 
                                 new_size=config['new_size'], 
                                 horizontal_flip=False, 
                                 train=train,
                                 discriminator_dataset=discriminator_dataset)
        
    elif domain == 'rotatedmnist':
        dataset = LabeledDataset(input_folder=config['data_path'], 
                                 keys=['0','1', '2', '3', '4', '5', '6', '7', '8', '9'], 
                                 domain='B', 
                                 rotate=False, 
                                 new_size=config['new_size'], 
                                 horizontal_flip=False, 
                                 train=train,
                                 discriminator_dataset=discriminator_dataset)
        
    elif domain == 'shoes_edges':
        dataset = LabeledDataset(input_folder=config['data_path'], 
                                 keys=['Shoes', 'Sandals', 'Slippers', 'Boots'], 
                                 domain='A', 
                                 rotate=False, 
                                 new_size=config['new_size'],
                                 train=train,
                                 discriminator_dataset=discriminator_dataset,
                                 low_pass=config['low_pass_domain1'] if 'low_pass_domain1' in config else False)
                                 
    elif domain == 'shoes':
        dataset = LabeledDataset(input_folder=config['data_path'], 
                                 keys=['Shoes', 'Sandals', 'Slippers', 'Boots'], 
                                 domain='B', 
                                 rotate=False, 
                                 new_size=config['new_size'],
                                 train=train,
                                 discriminator_dataset=discriminator_dataset)

                                 
    elif domain == 'rotatedshoes':
        dataset = LabeledDataset(input_folder=config['data_path'], 
                                 keys=['Shoes', 'Sandals', 'Slippers', 'Boots'], 
                                 domain='B', 
                                 rotate=True, 
                                 new_size=config['new_size'],
                                 train=train,
                                 discriminator_dataset=discriminator_dataset)
                                 
    elif domain == 'bitmoji':
        dataset = LabeledDataset(input_folder=config['data_path'], 
                                 keys=['Male', '~Male', 'Black_Hair', '~Black_Hair'] if 'keys' not in config else config['keys'],
                                 domain='B', 
                                 rotate=False, 
                                 new_size=config['new_size'],
                                 train=train,
                                 discriminator_dataset=discriminator_dataset,
                                 horizontal_flip=config['horizontal_flip'])
                                 
    elif domain== 'celebaForBitmoji':
        dataset = LabeledDataset(input_folder=config['data_path'], 
                                 keys=['Male', '~Male', 'Black_Hair', '~Black_Hair'] if 'keys' not in config else config['keys'],
                                 domain='A', 
                                 rotate=False, 
                                 new_size=config['new_size'],
                                 train=train,
                                 discriminator_dataset=discriminator_dataset,
                                 horizontal_flip=config['horizontal_flip'])

    elif domain== 'dummyA':
        dataset = LabeledDataset(input_folder=config['data_path'], 
                                keys=['attr'] if 'keys' not in config else config['keys'],
                                domain='A', 
                                rotate=False, 
                                new_size=config['new_size'],
                                train=train,
                                discriminator_dataset=discriminator_dataset,
                                horizontal_flip=config['horizontal_flip'],
                                include_negatives=config['include_negatives'] if 'include_negatives' in config else False)

    elif domain== 'dummyB':
        dataset = LabeledDataset(input_folder=config['data_path'], 
                                keys=['attr'] if 'keys' not in config else config['keys'],
                                domain='B', 
                                rotate=False, 
                                new_size=config['new_size'],
                                train=train,
                                discriminator_dataset=discriminator_dataset,
                                horizontal_flip=config['horizontal_flip'],
                                include_negatives=config['include_negatives'] if 'include_negatives' in config else False)
                                 
    else:
        print('domain {} not implemented'.format(domain))
        raise NotImplementedError
    return torch.utils.data.DataLoader(dataset=dataset,
                                        batch_size=config['batch_size'],
                                        shuffle=train,
                                        num_workers=config['num_workers'])
    
