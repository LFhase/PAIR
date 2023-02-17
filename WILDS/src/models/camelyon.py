import os
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import densenet121
from wilds.common.data_loaders import get_eval_loader
from wilds.datasets.camelyon17_dataset import Camelyon17Dataset

from .datasets import GeneralWilds_Batched_Dataset

IMG_HEIGHT = 224
NUM_CLASSES = 2

class Model(nn.Module):
    def __init__(self, args, weights):
        super(Model, self).__init__()
        self.num_classes = NUM_CLASSES
        self.enc = densenet121(pretrained=False).features # remove fc layer
        self.classifier = nn.Linear(1024, self.num_classes)
        if weights is not None:
            self.load_state_dict(deepcopy(weights))

    def reset_weights(self, weights):
        self.load_state_dict(deepcopy(weights))

    @staticmethod
    def getDataLoaders(args, device):
        full_dataset = Camelyon17Dataset(root_dir=os.path.join(args.data_dir, 'wilds'), download=True)

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        # get all train data
        train_data = full_dataset.get_subset('train', transform=transform)
        # separate into subsets by distribution
        train_sets = GeneralWilds_Batched_Dataset(train_data, args.batch_size, domain_idx=0, drop_last=not args.no_drop_last)
        # take subset of test and validation, making sure that only labels appeared in train
        # are included
        datasets = {}
        for split in full_dataset.split_dict:
            if split != 'train':
                datasets[split] = full_dataset.get_subset(split, transform=transform)

        # get the loaders
        kwargs = {'num_workers': args.num_workers, 'pin_memory': True, 'drop_last': False} \
            if device.type == "cuda" else {}
        train_loaders = DataLoader(train_sets, batch_size=args.batch_size, shuffle=True, **kwargs)
        
        kwargs = {'num_workers': args.num_workers, 'pin_memory': True, 'drop_last': False}
        tv_loaders = {}
        for split, dataset in datasets.items():
            tv_loaders[split] = get_eval_loader('standard', dataset, batch_size=256,**kwargs)
        return train_loaders, tv_loaders,full_dataset

    def forward(self, x, get_feat=False,frozen_mode=False):
        if frozen_mode:
            self.enc.eval()
            self.classifier.train()
            with torch.no_grad():
                features = self.enc(x)
                out = F.relu(features, inplace=True)
                out = F.adaptive_avg_pool2d(out, (1, 1))
                out = torch.flatten(out, 1)
        else:
            features = self.enc(x)
            out = F.relu(features, inplace=True)
            out = F.adaptive_avg_pool2d(out, (1, 1))
            out = torch.flatten(out, 1)
        pred = self.classifier(out)
        if get_feat:
            return pred, out
        return pred
