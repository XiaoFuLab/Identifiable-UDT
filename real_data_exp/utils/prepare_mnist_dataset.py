# load MNIST dataset and save them as jpg
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from tqdm import trange
import torch

ROOT_SRC = '../data/MNIST_raw/'
ROOT_DEST = '../data/MNIST'

if not os.path.exists(ROOT_DEST):
    os.makedirs(ROOT_DEST)

if not os.path.exists(ROOT_SRC):
    os.makedirs(ROOT_SRC)

# Download MNIST dataset
# Define the transform to normalize the data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Download and load the MNIST dataset
train_dataset = datasets.MNIST(root=ROOT_SRC, train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root=ROOT_SRC, train=False, download=True, transform=transform)

train_data = train_dataset.data
train_labels = train_dataset.targets
test_data = test_dataset.data
test_labels = test_dataset.targets

# Create dictionaries for training and test data
training_set = (train_data, train_labels)
test_set = (test_data, test_labels)

# Save the datasets as training.pt and test.pt
torch.save(training_set, os.path.join(ROOT_SRC, 'training.pt'))
torch.save(test_set, os.path.join(ROOT_SRC, 'test.pt'))

train_filepath = 'training.pt'
test_filepath = 'test.pt'

trainA_folderpath = 'trainA'
trainB_folderpath = 'trainB'
testA_folderpath = 'testA'
testB_folderpath = 'testB'

trainA_labelpath = 'trainA_attr.csv'
trainB_labelpath = 'trainB_attr.csv'
testA_labelpath = 'testA_attr.csv'
testB_labelpath = 'testB_attr.csv'

def prepare_mnist(input_file, output_folder, output_labelpath, rotate=False):
    # load mnist dataset
    data = torch.load(input_file)
    classes = data[1].numpy().copy()
    data = data[0].numpy()
    
    # rotate the data by 90 degrees if rotate is True
    if rotate:
        data = np.rot90(data, axes=(1, 2))
    
    # save the data as images
    data = np.expand_dims(data, axis=1)
    data = np.repeat(data, 3, axis=1)
    data = data.astype(np.uint8)
    data = np.transpose(data, (0, 2, 3, 1))

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    for i in trange(data.shape[0]):
        plt.imsave(os.path.join(output_folder, str(i)+'.jpg'), data[i])
    
    # save labels as 1, -1 for positive and negative for each class (digit)
    
    labels = np.ones((classes.shape[0], 10))*-1
    labels = labels.astype(np.int)
    labels[np.arange(classes.shape[0]), classes] = 1


    # save the labels as csv file use the first column for the filename
    # In the first line, print image_id,0,1,2,3,4,5,6,7,8,9
    # In the next lines, print image filename, label for each image
    # e.g.  0.jpg,1,-1,-1,-1,-1,-1,-1,-1,-1,-1
    #       1.jpg,-1,1,-1,-1,-1,-1,-1,-1,-1,-1

    with open(output_labelpath, 'w') as f:
        f.write('image_id,0,1,2,3,4,5,6,7,8,9\n')
        for i in trange(labels.shape[0]):
            f.write(str(i)+'.jpg,'+','.join(labels[i].astype(str))+'\n')

    print('saved %s' % output_folder)

prepare_mnist(os.path.join(ROOT_SRC, train_filepath), os.path.join(ROOT_DEST, trainA_folderpath), os.path.join(ROOT_DEST, trainA_labelpath), rotate=False)
prepare_mnist(os.path.join(ROOT_SRC, train_filepath), os.path.join(ROOT_DEST, trainB_folderpath), os.path.join(ROOT_DEST, trainB_labelpath), rotate=True)
prepare_mnist(os.path.join(ROOT_SRC, test_filepath), os.path.join(ROOT_DEST, testA_folderpath), os.path.join(ROOT_DEST, testA_labelpath), rotate=False)
prepare_mnist(os.path.join(ROOT_SRC, test_filepath), os.path.join(ROOT_DEST, testB_folderpath), os.path.join(ROOT_DEST, testB_labelpath), rotate=True)
