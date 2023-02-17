import numpy as np
import os
from copy import deepcopy

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from torchvision.models import resnet50
from wilds.common.data_loaders import get_eval_loader
try:
    from wilds.datasets.rxrx1_dataset import RxRx1Dataset
except Exception as e:
    print("RxRx1 Dataset not supported")

from .datasets import GeneralWilds_Batched_Dataset

IMG_HEIGHT = 224
NUM_CLASSES = 1139

def initialize_rxrx1_transform(is_training):
    def standardize(x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=(1, 2))
        std = x.std(dim=(1, 2))
        std[std == 0.] = 1.
        return TF.normalize(x, mean, std)
    t_standardize = transforms.Lambda(lambda x: standardize(x))

    angles = [0, 90, 180, 270]
    def random_rotation(x: torch.Tensor) -> torch.Tensor:
        angle = angles[torch.randint(low=0, high=len(angles), size=(1,))]
        if angle > 0:
            x = TF.rotate(x, angle)
        return x
    t_random_rotation = transforms.Lambda(lambda x: random_rotation(x))

    if is_training:
        transforms_ls = [
            t_random_rotation,
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            t_standardize,
        ]
    else:
        transforms_ls = [
            transforms.ToTensor(),
            t_standardize,
        ]
    transform = transforms.Compose(transforms_ls)
    return transform

class Model(nn.Module):
    def __init__(self, args, weights):
        super(Model, self).__init__()
        self.num_classes = NUM_CLASSES
        resnet = resnet50(pretrained=True)
        self.enc = nn.Sequential(*list(resnet.children())[:-1]) # remove fc layer
        self.fc = nn.Linear(2048, self.num_classes)
        if weights is not None:
            self.load_state_dict(deepcopy(weights))

    def reset_weights(self, weights):
        self.load_state_dict(deepcopy(weights))


    @staticmethod
    def getDataLoaders(args, device):
        dataset = RxRx1Dataset(root_dir=os.path.join(args.data_dir, 'wilds'), download=True)

        # initialize transform
        train_transform = initialize_rxrx1_transform(is_training=True)
        eval_transform = initialize_rxrx1_transform(is_training=False)

        # get all train data
        train_data = dataset.get_subset('train', transform=train_transform)

        # separate into subsets by distribution
        train_sets = GeneralWilds_Batched_Dataset(train_data, args.batch_size, domain_idx=1, drop_last=not args.no_drop_last)
        # take subset of test and validation, making sure that only labels appeared in train
        # are included
        datasets = {}
        for split in dataset.split_dict:
            if split != 'train':
                datasets[split] = dataset.get_subset(split, transform=eval_transform)

        # get the loaders
        kwargs = {'num_workers': args.num_workers, 'pin_memory': True, 'drop_last': False} \
            if device.type == "cuda" else {}
        train_loaders = DataLoader(train_sets, batch_size=args.batch_size, shuffle=True, **kwargs)
        tv_loaders = {}
        for split, sep_dataset in datasets.items():
            tv_loaders[split] = get_eval_loader('standard', sep_dataset, batch_size=256, num_workers=args.num_workers)
        return train_loaders, tv_loaders, dataset

    def forward(self, x,get_feat=False,frozen_mode=False):
        # x = x.expand(-1, 3, -1, -1)  # reshape MNIST from 1x32x32 => 3x32x32
        if len(x.shape) == 3:
            x.unsqueeze_(0)
        if frozen_mode:
            self.enc.eval()
            self.fc.train()
            with torch.no_grad():
                e = self.enc(x)
        else:
            e = self.enc(x)
        out = self.fc(e.squeeze(-1).squeeze(-1))
        if get_feat:
            return out, e.squeeze(-1).squeeze(-1)
        return out
