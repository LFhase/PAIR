import argparse
import copy
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from backpack import backpack, extend
from backpack.extensions import BatchGrad
from torch import autograd, nn, optim
from torchvision import datasets

from models import MLP, TopMLP
from mydatasets import coloredmnist
from utils import (EMA, GeneralizedCELoss, correct_pred, mean_accuracy,
                   mean_mse, mean_nll, mean_weight, pretty_print, validation,
                   validation_details)
from pair_alg import *


def IGA_penalty(envs_logits, envs_y, scale, lossf):
    
    grads = []
    grad_mean = 0
    for i in range(len(envs_logits)):

        loss = lossf(envs_logits[i], envs_y[i])
        grad0 = [val.view(-1) for val in autograd.grad(loss, scale, create_graph=True)]
        grad0 = torch.cat(grad0)
        grads.append(grad0)
        grad_mean += grad0 / len(envs_logits)

    grad_mean  = grad_mean.detach()

    train_penalty = 0 
    for i in range(len(grads)):
        train_penalty += torch.sum((grads[i] - grad_mean)**2) 

    return train_penalty 

def IRM_penalty(envs_logits, envs_y, scale, lossf):

    train_penalty = 0 
    for i in range(len(envs_logits)):
        loss = lossf(envs_logits[i], envs_y[i])
        grad0 = autograd.grad(loss, [scale], create_graph=True)[0]
        train_penalty += torch.sum(grad0**2)

    train_penalty /= len(envs_logits)

    return train_penalty

def GM_penalty(envs_logits, envs_y, scale, lossf):
    
    grads = []
    grad_mean = 0
    for i in range(len(envs_logits)):

        loss = lossf(envs_logits[i], envs_y[i])
        grad0 = [val.view(-1) for val in autograd.grad(loss, scale, create_graph=True)]
        grad0 = torch.cat(grad0)
        grads.append(grad0)

    train_penalty  = 0 
    for i in range(len(grads)-1):
        for j in range(i+1,len(grads)):
            train_penalty += -torch.sum(grads[i]*grads[j])
    
    return train_penalty


def rsc_train(mlp, topmlp, 
    steps, 
    envs, test_envs, 
    lossf, \
    penalty_anneal_iters, penalty_term_weight, anneal_val, \
    lr,l2_regularizer_weight, freeze_featurizer=False, verbose=True,eval_steps=1, hparams={}):
    
    if freeze_featurizer:
        optimizer = optimizer = optim.Adam( [var for var in topmlp.parameters()],  lr=lr) 
    else:
        optimizer = optimizer = optim.Adam([var for var in mlp.parameters()] + \
            [var for var in topmlp.parameters()],  lr=lr) 

    drop_f = (1 - hparams['rsc_f_drop_factor']) * 100
    drop_b = (1 - hparams['rsc_b_drop_factor']) * 100
    num_classes = 2
    logs = []
    for step in range(steps):
        # inputs
        all_x = torch.cat([envs[i]['images'] for i in range(len(envs))])
        all_y = torch.cat([envs[i]['labels'] for i in range(len(envs))])



        # one-hot labels
        all_o = torch.nn.functional.one_hot(all_y, num_classes)
        # features
        all_f = mlp(all_x)
        # predictions
        all_p = topmlp(all_f)

        if step < penalty_anneal_iters:
            loss = F.cross_entropy(all_p, all_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            # Equation (1): compute gradients with respect to representation
            all_g = autograd.grad((all_p * all_o).sum(), all_f)[0]

            # Equation (2): compute top-gradient-percentile mask
            percentiles = np.percentile(all_g.cpu(), drop_f, axis=1)
            percentiles = torch.Tensor(percentiles)
            percentiles = percentiles.unsqueeze(1).repeat(1, all_g.size(1))
            mask_f = all_g.lt(percentiles.cuda()).float()

            # Equation (3): mute top-gradient-percentile activations
            all_f_muted = all_f * mask_f

            # Equation (4): compute muted predictions
            all_p_muted = topmlp(all_f_muted)

            # Section 3.3: Batch Percentage
            all_s = F.softmax(all_p, dim=1)
            all_s_muted = F.softmax(all_p_muted, dim=1)
            changes = (all_s * all_o).sum(1) - (all_s_muted * all_o).sum(1)
            percentile = np.percentile(changes.detach().cpu(), drop_b)
            mask_b = changes.lt(percentile).float().view(-1, 1)
            mask = torch.logical_or(mask_f, mask_b).float()

            # Equations (3) and (4) again, this time mutting over examples
            all_p_muted_again = topmlp(all_f * mask)

            # Equation (5): update
            loss = F.cross_entropy(all_p_muted_again, all_y)
            #print(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        
        if step % eval_steps == 0:
            train_loss, train_acc, test_worst_loss, test_worst_acc = \
            validation(topmlp, mlp, envs, test_envs, lossf)
            log = [np.int32(step), train_loss, train_acc,\
                np.int32(0),test_worst_loss, test_worst_acc]
            logs.append(log)
            if verbose:
                pretty_print(*log)
    
    return (train_acc, train_loss, test_worst_acc, test_worst_loss), logs

def vrex_train(mlp, topmlp, steps, envs, test_envs, lossf, \
    penalty_anneal_iters, penalty_term_weight, anneal_val, \
    lr,l2_regularizer_weight,freeze_featurizer=False, eval_steps=5, verbose=True ):
    logs = []
    
    if freeze_featurizer:
        optimizer = optimizer = optim.Adam( [var for var in topmlp.parameters()],  lr=lr) 
        for param in mlp.parameters():
            param.requires_grad = False 
       
    else:
        optimizer = optimizer = optim.Adam([var for var in mlp.parameters()] + \
            [var for var in topmlp.parameters()],  lr=lr) 

    for step in range(steps):

        train_penalty = 0
        erm_losses = []
        for env in envs:
            logits = topmlp(mlp(env['images']))
            #lossf = mean_nll if flags.lossf == 'nll' else mean_mse 
            env['nll'] = lossf(logits, env['labels'])
            env['acc'] = mean_accuracy(logits, env['labels'])
            erm_losses.append(env['nll'])

        erm_losses = torch.stack(erm_losses)
        
        train_penalty = erm_losses.var()
        erm_loss = erm_losses.sum() 

        loss = erm_loss.clone()

        weight_norm = 0
        for w in [var for var in mlp.parameters()] + [var for var in topmlp.parameters()]:
            weight_norm += w.norm().pow(2)
        loss += l2_regularizer_weight * weight_norm

        penalty_weight = (penalty_term_weight if step >= penalty_anneal_iters else anneal_val)
        loss += penalty_weight * train_penalty
        if penalty_weight > 1.0:
            # Rescale the entire loss to keep gradients in a reasonable range
            loss /= penalty_weight
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        if step % eval_steps == 0:
            train_loss, train_acc, test_worst_loss, test_worst_acc = \
            validation(topmlp, mlp, envs, test_envs, lossf)
            log = [np.int32(step), train_loss, train_acc,\
                train_penalty.detach().cpu().numpy(),test_worst_loss, test_worst_acc]
            logs.append(log)
            if verbose:
                pretty_print(*log)
    
    return (train_acc, train_loss, test_worst_acc, test_worst_loss), logs

def iga_train(mlp, topmlp, steps, envs, test_envs, lossf, \
    penalty_anneal_iters, penalty_term_weight, anneal_val, \
    lr,l2_regularizer_weight,freeze_featurizer=False, verbose=True, eval_steps = 5, hparams={}):
    
    if freeze_featurizer:
        optimizer = optimizer = optim.Adam( [var for var in topmlp.parameters()],  lr=lr) 
       
    else:
        optimizer = optimizer = optim.Adam([var for var in mlp.parameters()] + \
            [var for var in topmlp.parameters()],  lr=lr) 

    
    logs = []
    for step in range(steps):
        train_penalty = 0
        envs_logits = []
        envs_y = []
        erm_loss = 0
        for env in envs:
            logits = topmlp(mlp(env['images']))
            env['nll'] = lossf(logits, env['labels'])
            env['acc'] = mean_accuracy(logits, env['labels'])
            envs_logits.append(logits)
            envs_y.append(env['labels'])
            erm_loss += env['nll']

        if freeze_featurizer:
            params = [var for var in topmlp.parameters()]
        else:
            params = [var for var in mlp.parameters()] + [var for var in topmlp.parameters()]
        train_penalty = IGA_penalty(envs_logits, envs_y, params, lossf)
    
        

        loss = erm_loss.clone()


        weight_norm = 0
        for w in [var for var in mlp.parameters()] + [var for var in topmlp.parameters()]:
            weight_norm += w.norm().pow(2)
        loss += l2_regularizer_weight * weight_norm


        penalty_weight = (penalty_term_weight 
                if step >= penalty_anneal_iters else anneal_val)
        loss += penalty_weight * train_penalty
        if penalty_weight > 1.0:
            # Rescale the entire loss to keep gradients in a reasonable range
            loss /= penalty_weight
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % eval_steps == 0:
            train_loss, train_acc, test_worst_loss, test_worst_acc = \
            validation(topmlp, mlp, envs, test_envs, lossf)
            log = [np.int32(step), train_loss, train_acc,\
                train_penalty.detach().cpu().numpy(),test_worst_loss, test_worst_acc]
            logs.append(log)
            if verbose:
                pretty_print(*log)
    return (train_acc, train_loss, test_worst_acc, test_worst_loss), logs

def dro_train(mlp, topmlp, steps, envs, test_envs, lossf, \
    penalty_anneal_iters, penalty_term_weight, anneal_val, \
    lr,l2_regularizer_weight,freeze_featurizer=False, verbose=True,eval_steps=5,hparams={}):
    
    if freeze_featurizer:
        optimizer = optimizer = optim.Adam( [var for var in topmlp.parameters()],  lr=lr) 
        for param in mlp.parameters():
            param.requires_grad = False 
       
    else:
        optimizer = optimizer = optim.Adam([var for var in mlp.parameters()] + \
            [var for var in topmlp.parameters()],  lr=lr) 
    
    logs = []
    for step in range(steps):
        train_penalty = 0
        envs_logits = []
        envs_y = []
        
        erm_losses = []
        
        for env in envs:
            logits = topmlp(mlp(env['images']))
            env['nll'] = lossf(logits, env['labels'])
            env['acc'] = mean_accuracy(logits, env['labels'])
            envs_logits.append(logits)
            envs_y.append(env['labels'])
            erm_losses.append(env['nll'])
        
        loss = max(erm_losses)
        
        weight_norm = 0
        for w in [var for var in mlp.parameters()] + [var for var in topmlp.parameters()]:
            weight_norm += w.norm().pow(2)
        loss += l2_regularizer_weight * weight_norm
  
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % eval_steps == 0:
            train_loss, train_acc, test_losses, test_acces = \
            validation_details(topmlp, mlp, envs, test_envs, lossf)

            log = [np.int32(step), train_loss, train_acc,\
                np.int32(0),*test_losses, *test_acces]
            logs.append(log)
            if verbose:
                pretty_print(*log)
    return (train_acc, train_loss, test_losses, test_acces), logs

def sd_train(mlp, topmlp, steps, envs, test_envs, lossf, \
    penalty_anneal_iters, penalty_term_weight, anneal_val, \
    lr,l2_regularizer_weight,freeze_featurizer=False, verbose=True,eval_steps=5, hparams={'lr_s2_decay':500}):
    if freeze_featurizer:
        optimizer = optimizer = optim.Adam( [var for var in topmlp.parameters()],  lr=lr) 
        for param in mlp.parameters():
            param.requires_grad = False 
       
    else:
        optimizer = optimizer = optim.Adam([var for var in mlp.parameters()] + \
            [var for var in topmlp.parameters()],  lr=lr) 
    logs = []
    for step in range(steps):

        train_penalty = 0
        erm_loss = 0
        for env in envs:
            logits = topmlp(mlp(env['images']))
            
            #lossf = mean_nll if lossf == 'nll' else mean_mse

            env['nll'] = lossf(logits, env['labels'])
            env['acc'] = mean_accuracy(logits, env['labels'])
        
            train_penalty += (logits**2).mean() 
            erm_loss += env['nll']
    

        loss = erm_loss.clone()


        weight_norm = 0
        for w in [var for var in mlp.parameters()] \
            +[var for var in topmlp.parameters()]:
            weight_norm += w.norm().pow(2)

        loss += l2_regularizer_weight * weight_norm


        penalty_weight = (penalty_term_weight 
                if step >= penalty_anneal_iters else anneal_val)
        loss += penalty_weight * train_penalty
        if penalty_weight > 1.0:
            # Rescale the entire loss to keep gradients in a reasonable range
            loss /= penalty_weight

        if penalty_anneal_iters > 0 and step >= penalty_anneal_iters:
            # using anneal, so decay lr
            loss /= hparams['lr_s2_decay']
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % eval_steps == 0:
            train_loss, train_acc, test_worst_loss, test_worst_acc = \
            validation(topmlp, mlp, envs, test_envs, lossf)
            log = [np.int32(step), train_loss, train_acc,\
                train_penalty.detach().cpu().numpy(),test_worst_loss, test_worst_acc]
            logs.append(log)
            if verbose:
                pretty_print(*log)
    
    return (train_acc, train_loss, test_worst_acc, test_worst_loss), logs

def irm_train(mlp, topmlp, steps, envs, test_envs, lossf, \
    penalty_anneal_iters, penalty_term_weight, anneal_val, \
    lr,l2_regularizer_weight, freeze_featurizer=False, verbose=True, eval_steps= 5,hparams={}):
    if freeze_featurizer:
        optimizer = optimizer = optim.Adam( [var for var in topmlp.parameters()],  lr=lr) 
        for param in mlp.parameters():
            param.requires_grad = False 
       
    else:
        optimizer = optimizer = optim.Adam([var for var in mlp.parameters()] + \
            [var for var in topmlp.parameters()],  lr=lr) 
    
    logs = []
    for step in range(steps):


        envs_logits = []
        envs_y = []
        erm_loss = 0
        scale = torch.tensor([1.])[0].cuda().requires_grad_() 
        for env in envs:
            logits = topmlp(mlp(env['images'])) * scale
            env['nll'] = lossf(logits, env['labels'])
            env['acc'] = mean_accuracy(logits, env['labels'])
            envs_logits.append(logits)
            envs_y.append(env['labels'])
            erm_loss += env['nll']
         
        train_penalty = IRM_penalty(envs_logits, envs_y,scale, lossf)

        loss = erm_loss.clone()


        weight_norm = 0
        for w in [var for var in mlp.parameters()] + [var for var in topmlp.parameters()]:
            weight_norm += w.norm().pow(2)

        loss += l2_regularizer_weight * weight_norm


        penalty_weight = (penalty_term_weight 
                if step >= penalty_anneal_iters else anneal_val)
        loss += penalty_weight * train_penalty
        if penalty_weight > 1.0:
            # Rescale the entire loss to keep gradients in a reasonable range
            loss /= penalty_weight
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % eval_steps == 0:
            train_loss, train_acc, test_worst_loss, test_worst_acc = \
            validation(topmlp, mlp, envs, test_envs, lossf)
            log = [np.int32(step), train_loss, train_acc,\
                train_penalty.detach().cpu().numpy(),test_worst_loss, test_worst_acc]
            logs.append(log)
            if verbose:
                pretty_print(*log)
    
    return (train_acc, train_loss, test_worst_acc, test_worst_loss), logs

def clove_train(mlp, topmlp, steps, envs, test_envs, lossf, \
    penalty_anneal_iters, penalty_term_weight, anneal_val, \
    lr,l2_regularizer_weight,freeze_featurizer=False, verbose=True,eval_steps=5,hparams={}):
    if freeze_featurizer:
        optimizer = optimizer = optim.Adam( [var for var in topmlp.parameters()],  lr=lr) 
        for param in mlp.parameters():
            param.requires_grad = False 
       
    else:
        optimizer = optimizer = optim.Adam([var for var in mlp.parameters()] + \
            [var for var in topmlp.parameters()],  lr=lr) 
    
    logs = []
    batch_size = hparams['batch_size'] if 'batch_size' in hparams else 512
    kernel_scale = hparams['kernel_scale'] if 'kernel_scale' in hparams else 0.4
    def mmce_penalty(logits, y, kernel='laplacian'):

        c = ~((logits.flatten() > 0) ^ (y.flatten()>0.5))
        c = c.detach().float()

        preds = F.sigmoid(logits).flatten()

        y_hat = (preds < 0.5).detach().bool()
        
        confidence = torch.ones(len(y_hat)).cuda()
        confidence[y_hat] = 1-preds[y_hat]
        confidence[~y_hat] = preds[~y_hat]

        k = (-(confidence.view(-1,1)-confidence).abs() / kernel_scale).exp()

        
        conf_diff = (c - confidence).view(-1,1)  * (c -confidence) 

        res = conf_diff * k

        return res.sum() / (len(logits)**2)

    pretty_print('step', 'train nll', 'train acc', 'train penalty', 'test acc')

    for step in range(steps):
        length = min(len(envs[0]['labels']), len(envs[1]['labels']))

        idx0 = np.arange(length)
        np.random.shuffle(idx0)
        idx1 = np.arange(length)
        np.random.shuffle(idx1)
        idx = [idx0, idx1]

        for i in range(length // batch_size):

            train_penalty = 0
            train_nll = 0
            train_acc = 0
            for j, env in enumerate(envs[0:2]):
                x, y = env['images'], env['labels']
                x_batch, y_batch = x[idx[j][i*batch_size:(i+1)*batch_size]], y[idx[j][i*batch_size:(i+1)*batch_size]]
                logits = topmlp(mlp(x_batch))
                nll = mean_nll(logits, y_batch)
                acc = mean_accuracy(logits, y_batch)
                mmce = mmce_penalty(logits, y_batch)
                train_penalty += mmce
                train_nll += nll 
                train_acc += acc 

        train_acc /=2
        

        weight_norm = torch.tensor(0.).cuda()
        for w in mlp.parameters():
            weight_norm += w.norm().pow(2)

        loss = train_nll.clone()
        penalty_weight = (penalty_term_weight 
                    if step >= penalty_anneal_iters else anneal_val)
        loss += penalty_weight * train_penalty
        if penalty_weight > 1.0:
            # Rescale the entire loss to keep gradients in a reasonable range
            loss /= penalty_weight

        optimizer.zero_grad()
        

        loss.backward()
        optimizer.step()
        

        if step % eval_steps == 0:
            train_loss, train_acc, test_worst_loss, test_worst_acc = \
            validation(topmlp, mlp, envs, test_envs, lossf)
            log = [np.int32(step), train_loss, train_acc,\
                train_penalty.detach().cpu().numpy(),test_worst_loss, test_worst_acc]
            logs.append(log)
            if verbose:
                pretty_print(*log)
        
    return (train_acc, train_loss, test_worst_acc, test_worst_loss), logs

def fishr_train(mlp, topmlp, steps, envs, test_envs, lossf, \
    penalty_anneal_iters, penalty_term_weight, anneal_val, \
    lr,l2_regularizer_weight,freeze_featurizer=False, verbose=True, eval_steps=5, hparams={}):
    
    def compute_grads_variance(features, labels, classifier):
        logits = classifier(features)
        loss = bce_extended(logits, labels)
        
        with backpack(BatchGrad()):
            loss.backward(
                inputs=list(classifier.parameters()), retain_graph=True, create_graph=True
            )

        dict_grads = OrderedDict(
            [
                (name, weights.grad_batch.clone().view(weights.grad_batch.size(0), -1))
                for name, weights in classifier.named_parameters()
            ]
        )
        dict_grads_variance = {}
        for name, _grads in dict_grads.items():
            grads = _grads * labels.size(0)  # multiply by batch size
            env_mean = grads.mean(dim=0, keepdim=True)

            dict_grads_variance[name] = (grads).pow(2).mean(dim=0)

        return dict_grads_variance

    def l2_between_grads_variance(cov_1, cov_2):
        assert len(cov_1) == len(cov_2)
        cov_1_values = [cov_1[key] for key in sorted(cov_1.keys())]
        cov_2_values = [cov_2[key] for key in sorted(cov_2.keys())]
        return (
            torch.cat(tuple([t.view(-1) for t in cov_1_values])) -
            torch.cat(tuple([t.view(-1) for t in cov_2_values]))
        ).pow(2).sum()
    
    if freeze_featurizer:
        optimizer = optimizer = optim.Adam( [var for var in topmlp.parameters()],  lr=lr) 
        for param in mlp.parameters():
            param.requires_grad = False 
    else:
        optimizer = optimizer = optim.Adam([var for var in mlp.parameters()] + \
            [var for var in topmlp.parameters()],  lr=lr) 
    logs = []
    
    bce_extended = extend(nn.BCEWithLogitsLoss())
    for step in range(steps):
        for edx, env in enumerate(envs):
            features = mlp(env['images'])
            logits = topmlp(features)
            env['nll'] = mean_nll(logits, env['labels'])
            env['acc'] = mean_accuracy(logits, env['labels'])
            if edx in [0, 1]:
                # True when the dataset is in training
                optimizer.zero_grad()
                env["grads_variance"] = compute_grads_variance(features, env['labels'], topmlp)

        train_nll = torch.stack([envs[0]['nll'], envs[1]['nll']]).sum()
        train_acc = torch.stack([envs[0]['acc'], envs[1]['acc']]).mean()

        weight_norm = torch.tensor(0.).cuda()
        for w in mlp.parameters():
            weight_norm += w.norm().pow(2)

        loss = train_nll.clone()
        loss += l2_regularizer_weight * weight_norm

        dict_grads_variance_averaged = OrderedDict(
            [
                (
                    name,
                    torch.stack([envs[0]["grads_variance"][name], envs[1]["grads_variance"][name]],
                                dim=0).mean(dim=0)
                ) for name in envs[0]["grads_variance"]
            ]
        )
        fishr_penalty = (
            l2_between_grads_variance(envs[0]["grads_variance"], dict_grads_variance_averaged) +
            l2_between_grads_variance(envs[1]["grads_variance"], dict_grads_variance_averaged)
        )
        train_penalty = fishr_penalty

        
        penalty_weight = (penalty_term_weight 
                    if step >= penalty_anneal_iters else anneal_val)
        loss += penalty_weight * train_penalty
        if penalty_weight > 1.0:
            # Rescale the entire loss to keep gradients in a reasonable range
            loss /= penalty_weight

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % eval_steps == 0:
            train_loss, train_acc, test_worst_loss, test_worst_acc = \
            validation(topmlp, mlp, envs, test_envs, lossf)
            log = [np.int32(step), train_loss, train_acc,\
                train_penalty.detach().cpu().numpy(),test_worst_loss, test_worst_acc]
            logs.append(log)
            if verbose:
                pretty_print(*log)
        
    return (train_acc, train_loss, test_worst_acc, test_worst_loss), logs

def gm_train(mlp, topmlp, steps, envs, test_envs, lossf, \
    penalty_anneal_iters, penalty_term_weight, anneal_val, \
    lr,l2_regularizer_weight, freeze_featurizer=False,  verbose=True, eval_steps=5, hparams={}):

    if freeze_featurizer:
        optimizer = optimizer = optim.Adam( [var for var in topmlp.parameters()],  lr=lr) 
    else:
        optimizer = optimizer = optim.Adam([var for var in mlp.parameters()] + \
            [var for var in topmlp.parameters()],  lr=lr) 
    logs = []
    for step in range(steps):

        train_penalty = 0
        envs_logits = []
        envs_y = []
        for env in envs:
            logits = topmlp(mlp(env['images']))
            env['nll'] = lossf(logits, env['labels'])
            env['acc'] = mean_accuracy(logits, env['labels'])
            envs_logits.append(logits)
            envs_y.append(env['labels'])

            
        
        train_penalty = GM_penalty(envs_logits, envs_y, [var for var in mlp.parameters()] + [var for var in topmlp.parameters()], lossf)

        erm_loss = (envs[0]['nll'] + envs[1]['nll'])
        

        loss = erm_loss.clone()


        weight_norm = 0
        for w in [var for var in mlp.parameters()] + [var for var in topmlp.parameters()]:
            weight_norm += w.norm().pow(2)

        loss += flags.l2_regularizer_weight * weight_norm


        penalty_weight = (flags.penalty_weight 
                if step >= flags.penalty_anneal_iters else flags.anneal_val)
        loss += penalty_weight * train_penalty
        if penalty_weight > 1.0:
            # Rescale the entire loss to keep gradients in a reasonable range
            loss /= penalty_weight
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % eval_steps == 0:
            train_loss, train_acc, test_worst_loss, test_worst_acc = \
            validation(topmlp, mlp, envs, test_envs, lossf)
            log = [np.int32(step), train_loss, train_acc,\
                train_penalty.detach().cpu().numpy(),test_worst_loss, test_worst_acc]
            logs.append(log)
            if verbose:
                pretty_print(*log)
    
    return (train_acc, train_loss, test_worst_acc, test_worst_loss), logs

def lff_train(mlp, topmlp, steps, envs, test_envs, lossf, \
    penalty_anneal_iters, penalty_term_weight, anneal_val, \
    lr,l2_regularizer_weight, freeze_featurizer=False, verbose=True,eval_steps=5, hparams={}):
    
    if freeze_featurizer is False:
        raise NotImplementedError

    x = torch.cat([envs[i]['images'] for i in range(len(envs))])
    y = torch.cat([envs[i]['labels'] for i in range(len(envs))])

    y = y.long().flatten()
    logs = []
    if penalty_anneal_iters > 0:
        optimizer = torch.optim.Adam([var for var in mlp.parameters()] \
            + [var for var in topmlp.parameters()],
            lr=lr, weight_decay=l2_regularizer_weight,)
        
        for step  in range(penalty_anneal_iters):
            logits = topmlp(mlp(x))
            loss = F.cross_entropy(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            train_penalty = torch.tensor([0]).cuda()[0]

            if step % 5 == 0:
                train_loss, train_acc, test_worst_loss, test_worst_acc = \
                validation(topmlp, mlp, envs, test_envs, lossf)
                log = [np.int32(step), train_loss, train_acc,\
                    train_penalty.detach().cpu().numpy(),test_worst_loss, test_worst_acc]
                logs.append(log)
                if verbose:
                    pretty_print(*log)


    _mlp = copy.deepcopy(mlp)
    _topmlp = copy.deepcopy(topmlp)
    
    model_b = torch.nn.Sequential(_mlp, _topmlp)

    model_d = torch.nn.Sequential(mlp, topmlp)

    optimizer_b = torch.optim.Adam(
        model_b.parameters(),
        lr=lr / 100,
        weight_decay=l2_regularizer_weight,
    )
    optimizer_d = torch.optim.Adam(
        model_d.parameters(),
        lr=lr / 100,
        weight_decay= l2_regularizer_weight,
    )
    lossf = nn.CrossEntropyLoss(reduction='mean')
    criterion = nn.CrossEntropyLoss(reduction='none')
    bias_criterion = GeneralizedCELoss(q = penalty_term_weight)
    
    sample_loss_ema_b = EMA(y.cpu().numpy(), alpha=0.7)
    sample_loss_ema_d = EMA(y.cpu().numpy(), alpha=0.7)

    index = np.arange(len(y))
    for step in range(penalty_anneal_iters, steps):
        
        logit_b = model_b(x)
        logit_d = model_d(x)

        loss_b = criterion(logit_b, y).cpu().detach()
        loss_d = criterion(logit_d, y).cpu().detach()
        
        sample_loss_ema_b.update(loss_b,index)
        sample_loss_ema_d.update(loss_d,index)
        
        loss_b = sample_loss_ema_b.parameter[index].clone().detach()
        loss_d = sample_loss_ema_d.parameter[index].clone().detach()

        # mnist target has one class, so I can do in this way.
        label_cpu = y.cpu()
        num_classes = 2
        for c in range(num_classes):
            class_index = np.where(label_cpu == c)[0]
            max_loss_b = sample_loss_ema_b.max_loss(c)
            max_loss_d = sample_loss_ema_d.max_loss(c)
            loss_b[class_index] /= max_loss_b
            loss_d[class_index] /= max_loss_d

        loss_weight = loss_b / (loss_b + loss_d + 1e-8)

        loss_b_update = bias_criterion(logit_b, y)
        loss_d_update = criterion(logit_d, y) * loss_weight.cuda()
        loss = loss_b_update.mean() + loss_d_update.mean()
        
        optimizer_b.zero_grad()
        optimizer_d.zero_grad()
        loss.backward()
        optimizer_b.step()
        optimizer_d.step()
    
        train_penalty = torch.tensor([0]).cuda()[0]

        if step % eval_steps == 0:
            train_loss, train_acc, test_worst_loss, test_worst_acc = \
            validation(topmlp, mlp, envs, test_envs, lossf)
            log = [np.int32(step), train_loss, train_acc,\
                train_penalty.detach().cpu().numpy(),test_worst_loss, test_worst_acc]
            logs.append(log)
            if verbose:
                pretty_print(*log)
    return (train_acc, train_loss, test_worst_acc, test_worst_loss), logs

def erm_train(mlp, topmlp, steps, envs, test_envs, lossf, \
    penalty_anneal_iters, penalty_term_weight, anneal_val, \
    lr,l2_regularizer_weight, freeze_featurizer=False, verbose=True,eval_steps=5, hparams={}):
    
    
    x = torch.cat([envs[i]['images'] for i in range(len(envs))])
    y = torch.cat([envs[i]['labels'] for i in range(len(envs))])

    if freeze_featurizer:
        optimizer = optimizer = optim.Adam( [var for var in topmlp.parameters()],  lr=lr) 
        for param in mlp.parameters():
            param.requires_grad = False 
        print('freeze_featurizer')
       
    else:
        optimizer = optimizer = optim.Adam([var for var in mlp.parameters()] + \
            [var for var in topmlp.parameters()],  lr=lr) 
    
    logs = []
    for step  in range(steps):
       
        logits = topmlp(mlp(x))
        #print(logits)
        loss = lossf(logits, y)
        #print(loss)
        #0/0
        weight_norm = 0
        for w in [var for var in mlp.parameters()] + [var for var in topmlp.parameters()]:
            weight_norm += w.norm().pow(2)

        loss += l2_regularizer_weight * weight_norm

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        train_penalty = torch.tensor([0]).cuda()[0]

        if step % eval_steps == 0:
            train_loss, train_acc, test_worst_loss, test_worst_acc = \
            validation(topmlp, mlp, envs, test_envs, lossf)
            log = [np.int32(step), train_loss, train_acc,\
                train_penalty.detach().cpu().numpy(),test_worst_loss, test_worst_acc]
            logs.append(log)
            if verbose:
                pretty_print(*log)
        
    return (train_acc, train_loss, test_worst_acc, test_worst_loss), logs

def syn_train(mlp, topmlp, steps, envs, test_envs, lossf, \
    penalty_anneal_iters, penalty_term_weight, anneal_val, \
    lr,l2_regularizer_weight, verbose=True,eval_steps=50, hparams={}):

    x = torch.cat([envs[i]['images'] for i in range(len(envs))])
    y = torch.cat([envs[i]['labels'] for i in range(len(envs))])

    optimizer = optim.Adam([var for var in mlp.parameters()] \
            +[var for var in topmlp.parameters()], lr=lr)
    logs = []
    ntasks =  hparams['ntasks']
    for step  in range(steps):
        logits = topmlp(mlp(x))

        per_logits_size = logits.shape[1] // ntasks
        per_y_size = y.shape[1] // ntasks
        loss = 0
        for i in range(ntasks):

            loss += lossf(logits[:, i*per_logits_size:(i+1)*per_logits_size],y[:,i*per_y_size:(i+1)*per_y_size])
        loss /= ntasks


        weight_norm = 0
        for w in [var for var in mlp.parameters()] + [var for var in topmlp.parameters()]:
            weight_norm += w.norm().pow(2)


        loss += l2_regularizer_weight * weight_norm

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        train_penalty = torch.tensor([0]).cuda()[0]


        if step % eval_steps == 0:
            
            with torch.no_grad():
                for j, env in enumerate(envs + test_envs):
                    logits = topmlp(mlp(env['images']))
                    loss =  0
                    acc = 0
                    for i in range(ntasks):
                        per_logits = logits[:, i*per_logits_size:(i+1)*per_logits_size]
                        
                        if j < len(envs): 
                            per_y = env['labels'][:,i*per_y_size:(i+1)*per_y_size] 
                        else:
                            per_y = env['labels']
                        loss += lossf(per_logits,per_y)
                        
                        acc += mean_accuracy(per_logits, per_y)
                    
                    loss /= ntasks
                    acc /=ntasks

                    env['nll'] = loss
                    env['acc'] = acc

            test_worst_loss = torch.stack([env['nll'] for env in test_envs]).max()
            test_worst_acc  = torch.stack([env['acc'] for env in test_envs]).min()
            train_loss = torch.stack([env['nll'] for env in envs]).mean()
            train_acc  = torch.stack([env['acc'] for env in envs]).mean()
            
            train_loss, train_acc, test_worst_loss, test_worst_acc = train_loss.detach().cpu().numpy(), \
                                                                        train_acc.detach().cpu().numpy(), \
                                                                        test_worst_loss.detach().cpu().numpy(),\
                                                                        test_worst_acc.detach().cpu().numpy()
            log = [np.int32(step), train_loss, train_acc,\
                train_penalty.detach().cpu().numpy(),test_worst_loss, test_worst_acc]
            logs.append(log)
            
            if verbose:
                pretty_print(*log)
    
    return (train_acc, train_loss, test_worst_acc, test_worst_loss), logs

def get_train_func(methods):
    assert methods in ['rsc', 'vrex', 'iga','sd','irm','clove','fishr','gm','lff','erm','dro','syn','pair']
    return eval("%s_train" % methods)
