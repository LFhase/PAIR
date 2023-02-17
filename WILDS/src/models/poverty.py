import os
from copy import deepcopy
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from wilds.common.data_loaders import get_eval_loader
from wilds.datasets.poverty_dataset import PovertyMapDataset

from .resnet_multispectral import ResNet18
from .datasets import Poverty_Batched_Dataset

IMG_HEIGHT = 224
NUM_CLASSES = 1

def initialize_poverty_train_transform():
    """Adapted from the Wilds library, available at: https://github.com/p-lambda/wilds"""
    transforms_ls = [
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.1),
        transforms.ToTensor()]
    rgb_transform = transforms.Compose(transforms_ls)

    def transform_rgb(img):
        # bgr to rgb and back to bgr
        img[:3] = rgb_transform(img[:3][[2,1,0]])[[2,1,0]]
        return img
    transform = transforms.Lambda(lambda x: transform_rgb(x))
    return transform
  

class Model(nn.Module):
    def __init__(self, args, weights):
        super(Model, self).__init__()
        self.num_classes = NUM_CLASSES

        self.enc = ResNet18(num_classes=1, num_channels=8)
        if weights is not None:
            self.load_state_dict(deepcopy(weights))

    def reset_weights(self, weights):
        self.load_state_dict(deepcopy(weights))

    

    @staticmethod
    def getDataLoaders(args, device):
        kwargs = {'no_nl': False,
                  'fold': ['A', 'B', 'C', 'D', 'E'][args.seed],
                #   'oracle_training_set': False,
                  'use_ood_val': True}
        dataset = PovertyMapDataset(root_dir=os.path.join(args.data_dir, 'wilds'),
                                    download=True, **kwargs)
        # get all train data
        # transform = initialize_poverty_train_transform()
        # In latest wilds example code, no transfromation is applied
        transform = transforms.Compose([])

        train_sets = Poverty_Batched_Dataset(dataset, 'train', args.batch_size, transform, drop_last=not args.no_drop_last)
        datasets = {}
        for split in dataset.split_dict:
            if split != 'train':
                datasets[split] = dataset.get_subset(split, transform=transform)
                print(split, len(datasets[split]))

        kwargs = {'num_workers': args.num_workers, 'pin_memory': True, 'drop_last': False, 'shuffle': True} \
            if device.type == "cuda" else {}
        train_loaders = DataLoader(train_sets, batch_size=args.batch_size, **kwargs)
        tv_loaders = {}
        for split, sep_dataset in datasets.items():
            tv_loaders[split] = get_eval_loader('standard', sep_dataset, batch_size=256, num_workers=args.num_workers)
        return train_loaders, tv_loaders, dataset

    def forward(self, x, get_feat=False,frozen_mode=False):
        if frozen_mode:
            self.enc.eval()
            self.enc.fc.train()
            with torch.no_grad():
                pred, feat = self.enc(x, with_feats=True)
            pred = self.enc.fc(feat)
            if get_feat:
                return pred, feat
            else:
                return pred
        return self.enc(x,with_feats=get_feat)
