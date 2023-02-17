import os
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import densenet121
from wilds.common.data_loaders import get_eval_loader
from wilds.datasets.fmow_dataset import FMoWDataset

from .datasets import FMoW_Batched_Dataset

IMG_HEIGHT = 224
NUM_CLASSES = 62

class Model(nn.Module):
    def __init__(self, args, weights):
        super(Model, self).__init__()
        self.num_classes = NUM_CLASSES
        self.enc = densenet121(pretrained=True).features
        self.classifier = nn.Linear(1024, self.num_classes)
        if weights is not None:
            self.load_state_dict(deepcopy(weights))

    def reset_weights(self, weights):
        self.load_state_dict(deepcopy(weights))

    @staticmethod
    def getDataLoaders(args, device):
        dataset = FMoWDataset(root_dir=os.path.join(args.data_dir, 'wilds'), download=True)
        # get all train data
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        train_sets = FMoW_Batched_Dataset(dataset, 'train', args.batch_size, transform, drop_last=not args.no_drop_last)
        datasets = {}
        for split in dataset.split_dict:
            if split != 'train':
                datasets[split] = dataset.get_subset(split, transform=transform)

        # get the loaders
        kwargs = {'num_workers': args.num_workers, 'pin_memory': True, 'drop_last': False} \
            if device.type == "cuda" else {}
        train_loaders = DataLoader(train_sets, batch_size=args.batch_size, shuffle=True, **kwargs)
        tv_loaders = {}
        for split, sep_dataset in datasets.items():
            tv_loaders[split] = get_eval_loader('standard', sep_dataset, batch_size=256, num_workers=args.num_workers)
        return train_loaders, tv_loaders, dataset

    def forward(self, x, get_feat=False,frozen_mode=False):
        
        if frozen_mode:
            self.enc.eval()
            self.classifier.train()
            with torch.no_grad():
                features = self.enc(x)
                out = F.relu(features, inplace=True)
                out = F.adaptive_avg_pool2d(out, (1, 1))
                out_features = torch.flatten(out, 1)
        else:
            features = self.enc(x)
            out = F.relu(features, inplace=True)
            out = F.adaptive_avg_pool2d(out, (1, 1))
            out_features = torch.flatten(out, 1)
        out = self.classifier(out_features)
        if get_feat:
            return out, out_features
        return out
