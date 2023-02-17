import argparse
import copy
import os
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
from train import get_train_func
from utils import (EMA, GeneralizedCELoss, correct_pred, mean_accuracy,
                   mean_mse, mean_nll, mean_weight, parse_bool, pretty_print,
                   validation)


def main(flags):
    if flags.save_dir is not None and not os.path.exists(flags.save_dir):
        os.makedirs(flags.save_dir)
    flags.freeze_featurizer = False if flags.freeze_featurizer.lower() == 'false' else True 
    final_train_accs = []
    final_train_losses = []
    final_test_accs = []
    final_test_losses = []
    final_grad_sims = []
    logs = []


    for restart in range(flags.n_restarts):
        if flags.seed>=0 and restart != flags.seed:
            print(f"Jump over seed {restart}")
            continue
        if flags.verbose:
            print("Restart", restart)
        

        ### loss function binary_cross_entropy 
        input_dim = 2 * 14 * 14
        if flags.methods in ['rsc', 'lff']:
            n_targets = 2
            lossf = F.cross_entropy
            int_target = True 
        else:
            n_targets = 1 
            lossf = mean_nll 
            int_target = False  


        np.random.seed(restart)
        torch.manual_seed(restart)
        ### load datasets 
        if flags.dataset == 'coloredmnist025' or flags.dataset == 'coloredmnist25':
            envs, test_envs = coloredmnist(0.25, 0.1, 0.2, int_target = int_target)
        elif flags.dataset == 'coloredmnist025gray':
            envs, test_envs = coloredmnist(0.25, 0.5, 0.5,int_target = int_target)
        elif flags.dataset == 'coloredmnist01':
            envs, test_envs = coloredmnist(0.1, 0.2, 0.25, int_target = int_target)
        elif flags.dataset == 'coloredmnist01gray':
            envs, test_envs = coloredmnist(0.1, 0.5, 0.5,  int_target = int_target)
        elif flags.dataset == 'coloredmnist':
            envs, test_envs = coloredmnist(flags.flip_p,flags.envs_p[0],flags.envs_p[1],  int_target = int_target)
        else:
            raise NotImplementedError
        

        mlp = MLP(hidden_dim = flags.hidden_dim, input_dim=input_dim).cuda()
        topmlp = TopMLP(hidden_dim = flags.hidden_dim, n_top_layers=flags.n_top_layers, \
            n_targets=n_targets, fishr= flags.methods in ['fishr']).cuda()

        print(mlp, topmlp)

        if flags.load_model_dir is not None and os.path.exists(flags.load_model_dir):
            device = torch.device("cuda")
            state = torch.load(os.path.join(flags.load_model_dir,'mlp%d.pth' % restart), map_location=device)
            mlp.load_state_dict(state)
            
            state = torch.load(os.path.join(flags.load_model_dir,'topmlp%d.pth' % restart), map_location=device)
            topmlp.load_state_dict(state)
            print("Load model from %s" % flags.load_model_dir)
            

        if len(flags.group_dirs)>0:
            print('load groups')
            x = torch.cat([env['images'] for env in envs])
            y = torch.cat([env['labels'] for env in envs])
            #print(x.shape, y.shape)
            groups = [np.load(os.path.join(group_dir,'group%d.npy' % restart)) for group_dir in flags.group_dirs]
            n_groups = len(groups)
            new_envs = []

            for group in groups:
                for val in np.unique(group):
                    env = {}
                    env['images'] = x[group == val]
                    env['labels'] = y[group == val]

                    new_envs.append(env)
            train_envs = new_envs

        else:
            train_envs = envs

        train_func = get_train_func(flags.methods)
        params = [mlp, topmlp, flags.steps, train_envs, test_envs,lossf,\
            flags.penalty_anneal_iters, flags.penalty_weight, \
            flags.anneal_val, flags.lr, \
            flags.l2_regularizer_weight, flags.freeze_featurizer, flags.eval_steps, flags.verbose, ]
        if flags.methods in ['vrex', 'iga','irm','fishr','gm','lff','erm','dro','pair']:
            hparams = {}
        elif flags.methods in ['clove']:
            hparams = {'batch_size': flags.batch_size, 'kernel_scale': flags.kernel_scale}
        elif flags.methods in ['rsc']:
            hparams = {'rsc_f_drop_factor' : flags.rsc_f, 'rsc_b_drop_factor': flags.rsc_b}
        elif flags.methods in ['sd']:
            hparams = {'lr_s2_decay': flags.lr_s2_decay}
        else:
            raise NotImplementedError
        # additional exp configs
        hparams['opt'] = flags.opt
        # hparams['pair_bal'] = flags.pair_bal
        hparams['opt_verbose'] = flags.opt_verbose
        hparams['opt_coe'] = flags.opt_coe
        hparams['pair_sim'] = flags.pair_sim

        res = train_func(*params,hparams)
        (train_acc, train_loss, test_worst_acc, test_worst_loss), per_logs = res 
        
        
        logs.extend(per_logs)
        final_train_accs.append(train_acc)
        final_train_losses.append(train_loss)
        final_test_accs.append(test_worst_acc)
        final_test_losses.append(test_worst_loss)

        if flags.verbose:
            
            print('Final train acc (mean/std across restarts so far):')
            print(np.mean(final_train_accs), np.std(final_train_accs))
            print('Final train loss (mean/std across restarts so far):')
            print(np.mean(final_train_losses), np.std(final_train_losses))
            print('Final worest test acc (mean/std across restarts so far):')
            print(np.mean(final_test_accs), np.std(final_test_accs))
            print('Final worest test loss (mean/std across restarts so far):')
            print(np.mean(final_test_losses), np.std(final_test_losses))

        results = [np.mean(final_train_accs), np.std(final_train_accs), 
                                np.mean(final_train_losses), np.std(final_train_losses), 
                                np.mean(final_test_accs), np.std(final_test_accs), 
                                np.mean(final_test_losses), np.std(final_test_losses), 
                                ]
            
    

        if flags.save_dir is not None:
            state = mlp.state_dict()
            torch.save(state, os.path.join(flags.save_dir,'mlp%d.pth' % restart))
            state = topmlp.state_dict()
            torch.save(state, os.path.join(flags.save_dir,'topmlp%d.pth'  % restart))
            
            with torch.no_grad():
                x = torch.cat([env['images'] for env in envs])
                y = torch.cat([env['labels'] for env in envs])
                logits = topmlp(mlp(x))
            group, _ = correct_pred(logits, y)

            pseudolabel = np.copy(y.cpu().numpy().flatten())
            pseudolabel[~group] = 1-pseudolabel[~group]
            np.save(os.path.join(flags.save_dir,'group%d.npy' % restart), group)
            np.save(os.path.join(flags.save_dir,'pseudolabel%d.npy' % restart), pseudolabel )

    logs = np.array(logs)
    
    if flags.save_dir is not None:
        np.save(os.path.join(flags.save_dir,'logs.npy'), logs)

    return results, logs

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Colored MNIST & CowCamel')
    parser.add_argument('--verbose', type=bool, default=False)
    # additional exp name id
    parser.add_argument('--exp_id', type=str,default='default')
    parser.add_argument('--n_restarts', type=int, default=10)
    parser.add_argument('--dataset', type=str, default='coloredmnist025')
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--n_top_layers', type=int, default=1)
    parser.add_argument('--l2_regularizer_weight', '-l2',type=float,default=0.001)
    
    parser.add_argument('--opt', type=str, default='adam')
    parser.add_argument('--opt_verbose',action='store_true')
    parser.add_argument('--pair_sim',action='store_true')
    parser.add_argument('--opt_coe', type=float, default=0.0)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--steps', type=int, default=501)
    parser.add_argument('--lossf', type=str, default='nll')
    parser.add_argument('--penalty_anneal_iters', '-pi', type=int, default=100)
    parser.add_argument('--penalty_weight', '-p', type=float, default=10000.0)
    parser.add_argument('--irmx_p2', '-p2', type=float, default=-1)
    parser.add_argument('--anneal_val', '-av',type=float, default=1)
    
    parser.add_argument('--methods', type=str, default='irmv2')
    parser.add_argument('--lr_s2_decay', type=float, default=500)
    parser.add_argument('--freeze_featurizer', type=str, default='False')
    parser.add_argument('--eval_steps', type=int, default=5)
    parser.add_argument('--seed', type=int, default=-1) # eval at a specific seed
    
    parser.add_argument('--load_model_dir', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--group_dirs', type=str, nargs='*',default={})
    
    #RSC
    parser.add_argument('--rsc_f', type=float, default=0.99)
    parser.add_argument('--rsc_b', type=float, default=0.97)

    #clove 
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--kernel_scale', type=float, default=0.4)

    parser.add_argument('--n_examples', type=int, default=18000)

    parser.add_argument('--flip_p', default=0.25, type=float)
    parser.add_argument('--envs_p', nargs='?', default='[0.1,0.2]', help='random seed')
    parser.add_argument('--norun',type=parse_bool, default=False)
    
    parser.add_argument('--no_plot',action='store_true')
    flags = parser.parse_args()
    flags.envs_p = eval(flags.envs_p)
    if flags.norun:
        if flags.verbose:
            print('Flags:')
            for k,v in sorted(vars(flags).items()):
                print("\t{}: {}".format(k, v))
    else:   
        main(flags)






