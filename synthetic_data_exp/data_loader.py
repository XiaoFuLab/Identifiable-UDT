import torch
from torch.utils.data.dataset import Dataset
import pickle
from invertible_network_utils import *


class ViewDatasetWithLabels(Dataset):
    def __init__(self, v1, v2, l1, l2):
        self.v1 = v1
        self.v2 = v2
        self.data_len = v1.shape[0]
        self.l1 = l1
        self.l2 = l2

    def __getitem__(self, index):
        return self.v1[index], self.l1[index], self.v2[index], self.l2[index]
    
    def __len__(self):
        return self.data_len
    
def get_loaders_from_file(dtype=torch.float32, 
                        batch_size=1000, 
                        shuffle=True,
                        save_filename=None):
    
    with open(save_filename, 'rb') as handle:
        data = pickle.load(handle)

    view1 = data['view1_train']
    view2 = data['view2_train']
    class_labels = data['class_train']
    view1_test = data['view1_test']
    view2_test = data['view2_test']
    class_labels_test = data['class_test']

    # permute the train samples so that they are no longer aligned
    perm_idx = torch.randperm(view1.size()[0])
    view2 = view2[perm_idx]
    class_labels_view2 = class_labels[perm_idx]
    
    dataset_train = ViewDatasetWithLabels(view1, view2, class_labels, class_labels_view2)
    dataset_test = ViewDatasetWithLabels(view1_test, view2_test, class_labels_test, class_labels_test)
    
    train_loader = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=shuffle)
    test_loader = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=shuffle)
    
    return train_loader, test_loader


def get_all_loaders_gaussian(dtype=torch.float32, 
                             batch_size=1000, 
                             num_train_samples=10000, 
                             num_test_samples=1000, 
                             shuffle=True,
                             num_components=2,
                             num_conditionals=None,
                             dim=2,
                             save_filename=None,
                             variance=0.3,
                             cond_thresh_ratio=0.5):
    

    means = [(torch.rand(dim)-0.5)*2.0 for _ in range(num_components)]
    variances = [torch.ones((dim,))*variance for _ in range(num_components)]
    
    print('Generating data...') 
    view1 = []
    view1_test = []
    for i in range(len(means)):
        view1.append(torch.randn(num_train_samples//num_components, dim)*variances[i]+means[i])
        view1_test.append(torch.randn(num_test_samples//num_components, dim)*variances[i]+means[i])
        
    view1 = torch.cat(view1, dim=0).type(dtype)
    view1_test = torch.cat(view1_test, dim=0).type(dtype)
    
    # Generate class labels (labels corresponding to conditional distributions)
    if num_conditionals is None:
        num_conditionals = num_components
    class_labels = torch.zeros((num_train_samples, num_conditionals))
    for i in range(num_components):
        #compute the index such that 
        class_labels[i*num_train_samples//num_components:(i+1)*num_train_samples//num_components, int(i*num_conditionals/num_components)] = 1
    
    class_labels_test = torch.zeros((num_test_samples, num_components))
    for i in range(num_components):
        class_labels_test[i*num_test_samples//num_components:(i+1)*num_test_samples//num_components, int(i*num_conditionals/num_components)] = 1
        
    translation_func = invertible_network_hyvarinen(
                                            n=view1.shape[1],
                                            n_layers=3,
                                            cond_thresh_ratio=cond_thresh_ratio,
                                            n_iter_cond_thresh=25000,
                                            act_fct='smooth_leaky_relu')
    view2 = translation_func(view1)
    view2_test = translation_func(view1_test)
    print('Generated data.')
    # Save the data so that we don't have to retrain.
    if save_filename is not None:
        import pickle
        from pickle import HIGHEST_PROTOCOL
        a = {'view1_train':view1, 
            'view2_train':view2,
            'class_train': class_labels,
            'view1_test':view1_test,
            'view2_test':view2_test,
            'class_test': class_labels_test}
        with open(save_filename, 'wb') as handle:
            pickle.dump(a, handle, protocol=HIGHEST_PROTOCOL)
        print('successfully saved into file {}'.format(save_filename))
    
    # permute the train samples so that they are no longer aligned
    perm_idx = torch.randperm(view1.size()[0])
    view2 = view2[perm_idx]
    class_labels_view2 = class_labels[perm_idx]
    
    dataset_train = ViewDatasetWithLabels(view1, view2, class_labels, class_labels_view2)
    dataset_test = ViewDatasetWithLabels(view1_test, view2_test, class_labels_test, class_labels_test)
    
    train_loader = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=shuffle)
    test_loader = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=shuffle)
    
    return train_loader, test_loader
