import os
from copy import deepcopy

import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet50
from wilds.common.data_loaders import get_eval_loader
from wilds.datasets.iwildcam_dataset import IWildCamDataset

from .datasets import GeneralWilds_Batched_Dataset
import torch

IMG_HEIGHT = 224
NUM_CLASSES = 186

def get_image_base_transform_steps(dataset, target_resolution=None):
    transform_steps = []

    if dataset.original_resolution is not None and min(
        dataset.original_resolution
    ) != max(dataset.original_resolution):
        crop_size = min(dataset.original_resolution)
        transform_steps.append(transforms.CenterCrop(crop_size))

    if target_resolution is not None:
        transform_steps.append(transforms.Resize(target_resolution))

    return transform_steps

class Model(nn.Module):
    def __init__(self, args, weights):
        super(Model, self).__init__()
        self.num_classes = NUM_CLASSES
        pretrain_path=os.path.join(args.data_dir,'wilds',args.dataset)
        if os.path.exists(pretrain_path):
            resnet = resnet50(pretrained=False)
            resnet.load_state_dict(torch.load(pretrain_path + f'/resnet50.rar'))
            print(f"Load pretrained resnet from {pretrain_path}")
        else:
            resnet = resnet50(pretrained=True)
            if not os.path.exists(pretrain_path):
                os.makedirs(pretrain_path)
            torch.save(resnet.state_dict(),pretrain_path+ f'/resnet50.rar')
            print(f"Load pretrained resnet from url")
        self.enc = nn.Sequential(*list(resnet.children())[:-1]) # remove fc layer
        self.fc = nn.Linear(2048, self.num_classes)
        if weights is not None:
            self.load_state_dict(deepcopy(weights))

    def reset_weights(self, weights):
        self.load_state_dict(deepcopy(weights))


    @staticmethod
    def getDataLoaders(args, device):
        dataset = IWildCamDataset(root_dir=os.path.join(args.data_dir, 'wilds'), download=True)
        # get all train data
        transform = transforms.Compose([
            # transforms.Resize((224, 224)),
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        train_data = dataset.get_subset('train', transform=transform)
        # separate into subsets by distribution
        train_sets = GeneralWilds_Batched_Dataset(train_data, args.batch_size, domain_idx=0, drop_last=not args.no_drop_last)
        # take subset of test and validation, making sure that only labels appeared in train
        # are included
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
