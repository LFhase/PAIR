
import numpy as np
import torch
from torchvision import datasets
import math
import os
import torch
from PIL import Image, ImageFile
from torchvision import transforms
import torchvision.datasets.folder
from torch.utils.data import TensorDataset, Subset
from torchvision.datasets import MNIST, ImageFolder
from torchvision.transforms.functional import rotate


from misc import split_dataset,make_weights_for_balanced_classes,seed_hash
# from fast_data_loader import InfiniteDataLoader, FastDataLoader


#coloredmnist are modified from https://github.com/facebookresearch/InvariantRiskMinimization
def coloredmnist(label_noise_rate, trenv1, trenv2, int_target=False):
    # Load MNIST, make train/val splits, and shuffle train set examples
    mnist = datasets.MNIST('./data/MNIST', train=True, download=True)
    mnist_train = (mnist.data[:50000], mnist.targets[:50000])
    mnist_val = (mnist.data[50000:], mnist.targets[50000:])
    
    rng_state = np.random.get_state()
    np.random.shuffle(mnist_train[0].numpy())
    np.random.set_state(rng_state)
    np.random.shuffle(mnist_train[1].numpy())

    # Build environments
    def make_environment(images, labels, e):
        def torch_bernoulli(p, size):
            return (torch.rand(size) < p).float()
        def torch_xor(a, b):
            return (a-b).abs() # Assumes both inputs are either 0 or 1
        # 2x subsample for computational convenience
        images = images.reshape((-1, 28, 28))[:, ::2, ::2]
        # Assign a binary label based on the digit; flip label with probability 0.25
        labels = (labels < 5).float()
        labels = torch_xor(labels, torch_bernoulli(label_noise_rate, len(labels)))
        # Assign a color based on the label; flip the color with probability e
        colors = torch_xor(labels, torch_bernoulli(e, len(labels)))
        # Apply the color to the image by zeroing out the other color channel
        images = torch.stack([images, images], dim=1)
        images[torch.tensor(range(len(images))), (1-colors).long(), :, :] *= 0
        
        if int_target:
            return {
                'images': (images.float() / 255.).cuda(), 
                'labels': labels[:, None].long().flatten().cuda()
            }
        else:
            return {
                'images': (images.float() / 255.).cuda(), 
                'labels': labels[:, None].cuda()
            }
             

    envs = [
        make_environment(mnist_train[0][::2], mnist_train[1][::2], trenv1),
        make_environment(mnist_train[0][1::2], mnist_train[1][1::2], trenv2)]
    
    # init 3 test environments [0.1, 0.5, 0.9] 
    test_envs = [    
        make_environment(mnist_val[0], mnist_val[1], 0.9),
        make_environment(mnist_val[0], mnist_val[1], 0.1),
        make_environment(mnist_val[0], mnist_val[1], 0.5),
    ]
    return envs, test_envs
