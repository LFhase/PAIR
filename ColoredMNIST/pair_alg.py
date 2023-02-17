import argparse
import copy
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from backpack import backpack, extend
from backpack.extensions import BatchGrad
from torch import autograd, nn, optim
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets

from misc import MovingAverage
from models import MLP, Net, TopMLP
from mydatasets import coloredmnist
from utils import (EMA, GeneralizedCELoss, correct_pred, mean_accuracy,
                   mean_mse, mean_nll, mean_weight, pretty_print, validation,
                   validation_details)
from pair import PAIR


def IRM_penalty_pair(envs_logits, envs_y, scale, lossf):

    train_penalty = 0 
    for i in range(len(envs_logits)):
        loss = lossf(envs_logits[i], envs_y[i])
        grad0 = autograd.grad(loss, [scale], create_graph=True)[0]
        train_penalty += torch.sum(grad0**2)

    train_penalty /= len(envs_logits)

    return train_penalty

def IRM_penalty_single(env_logits, env_y, scale, lossf):

    loss = lossf(env_logits*scale, env_y)
    grad0 = autograd.grad(loss, [scale], create_graph=True)[0]
    train_penalty = torch.sum(grad0**2)

    train_penalty /= len(env_logits)

    return train_penalty

def pair_train(mlp, topmlp, steps, envs, test_envs, lossf, \
    penalty_anneal_iters, penalty_term_weight, anneal_val, \
    lr,l2_regularizer_weight, freeze_featurizer=False, eval_steps= 5, verbose=True,hparams={}):
    net = Net(mlp,topmlp)
    if freeze_featurizer:
        trainable_params = [var for var in mlp.parameters()]
        for param in mlp.parameters():
            param.requires_grad = False 
    else:
        trainable_params = [var for var in mlp.parameters()] + \
            [var for var in topmlp.parameters()]
            
    if hparams['opt'].lower() == 'sgd':
        optimizer = optim.SGD(trainable_params,  lr=lr) 
    elif hparams['opt'].lower() == 'pair':
        optimizer = optim.Adam( trainable_params,  lr=1e-3) 
    else:
        optimizer = optim.Adam( trainable_params,  lr=lr) 
    
    logs = []
    for step in range(steps):
        envs_logits = []
        envs_y = []
        erm_losses = []
        scale = torch.tensor([1.])[0].cuda().requires_grad_() 
        for env in envs:
            logits = topmlp(mlp(env['images'])) * scale
            env['nll'] = lossf(logits, env['labels'])
            env['acc'] = mean_accuracy(logits, env['labels'])
            envs_logits.append(logits)
            envs_y.append(env['labels'])
            erm_losses.append(env['nll'])
         
        irm_penalty = IRM_penalty_pair(envs_logits, envs_y,scale, lossf)
        erm_losses = torch.stack(erm_losses)
        vrex_penalty = erm_losses.var()
        erm_loss = erm_losses.mean() 
        alphas = np.array([0])
        device = logits.device

        # Compile loss
        losses = torch.stack([erm_loss,irm_penalty,vrex_penalty]).to(device)

        if step >= penalty_anneal_iters:
            if step==penalty_anneal_iters:
                r = 1e-12
                r2 = 1e10
                r_l2 = r*r2
                preference = np.array([r]+[r_l2,(1-r-r_l2)])
                inner_optimizer = optim.SGD(trainable_params,  lr=lr,momentum=0.9) 
                optimizer = PAIR(topmlp.parameters(),inner_optimizer,preference=preference,eps=1e-1,verbose=hparams['opt_verbose'],coe=hparams['opt_coe'])
                print(f"Switch optimizer to {optimizer}")
            optimizer.zero_grad()
            optimizer.set_losses(losses=losses)
            pair_loss, moo_losses, mu_rl, alphas = optimizer.step()
            pair_res = np.array([pair_loss, mu_rl, alphas])
        else:
            loss = erm_loss
        
            weight_norm = 0
            for w in [var for var in mlp.parameters()] + [var for var in topmlp.parameters()]:
                weight_norm += w.norm().pow(2)
            penalty_weight = (penalty_term_weight 
                    if step >= penalty_anneal_iters else anneal_val)
            if penalty_weight > 1.0:
                # Rescale the entire loss to keep gradients in a reasonable range
                weight_norm /= penalty_weight
                
            loss += l2_regularizer_weight * weight_norm

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if step % eval_steps == 0:
            train_loss, train_acc, test_worst_loss, test_worst_acc = \
            validation(topmlp, mlp, envs, test_envs, lossf)
            
            print_log = [np.int32(step), train_loss, train_acc, \
                losses.detach().cpu().numpy(),alphas,test_worst_loss, test_worst_acc]
            log = [np.int32(step), train_loss, train_acc,\
                losses.detach().cpu().numpy(),test_worst_loss, test_worst_acc]
            logs.append(log)
            if verbose:
                pretty_print(*print_log)
        
    return (train_acc, train_loss, test_worst_acc, test_worst_loss), logs

