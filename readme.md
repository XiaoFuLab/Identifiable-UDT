# Towards Identifiable Unsupervised Domain Translation: A Diversified Distribution Matching Approach

Implementation of the paper "Towards Identifiable Unsupervised Domain Translation: A Diversified Distribution Matching Approach", ICLR 2024. [Arxiv](https://arxiv.org/abs/2401.09671)

## Installation

Install the required python packages listed in `requirements.txt`. If you use pip run the following:
```
pip install requirements.txt
```

## Data Preparation
The dataset folder needs to be in the following structure

- celebahq2bitmoji /
    - trainA /
        - image1.jpg
        - image2.jpg
        - ...
    - trainB /
    - testA /
    - testB /
    - trainA_attr.csv
    - trainB_attr.csv
    - testA_attr.csv
    - testB_attr.csv

For demo, download and prepare MNIST to rotated MNIST dataset with the following command
```
cd real_data_exp
python utils/prepare_mnist_dataset.py
```
It will download the MNIST dataset in the folder data/

## Synthetic Data Training 
For synthetic data training, the above data preparation is not required.
One can simply go to the `synthetic_data_exp` folder, open the notebook `demo.ipynb` and run the code there.

## Real Data Training 
For real data training, follow the data preparation step.
A demo script for MNIST is provided. Simply run
```
cd real_data_exp
bash scripts/mnist_demo.sh
```
One needs to specify the `model_path`, the location where model is to be saved, and `data_path` containing the training data in the config file `configs/mnist_demo.yaml`.


## Testing the trained model
In order to test the trained model, run `eval.py` and provide `checkpoint_path` (trained model checkpoint), and `dest_dir` (location for saving the image results). The test data is loaded from `data_path/testA` and `data_path/testB`. Note that auxiliary information is not required during test phase. A demo eval script for MNIST can be run using
```
cd real_dat_exp
bash scripts/test_mnist_demo.sh
```

Under construction ...
