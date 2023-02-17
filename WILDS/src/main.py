import copy
import argparse
import datetime
import json
import os
from statistics import mode
import sys
import csv
from tokenize import group
import tqdm
from collections import defaultdict
from tempfile import mkdtemp

import numpy as np
import torch
import torch.optim as optim
from scheduler import initialize_scheduler

import models
from config import dataset_defaults
from utils import get_preference, set_seed, unpack_data, sample_domains, save_best_model, \
    Logger, return_predict_fn, return_criterion, fish_step

# This is secret and shouldn't be checked into version control
os.environ["WANDB_API_KEY"]=None
# Name and notes optional
# WANDB_NAME="My first run"
# WANDB_NOTES="Smaller learning rate, more regularization."
import wandb
import traceback
from pair import PAIR
from torch.autograd import Variable


runId = datetime.datetime.now().isoformat().replace(':', '_')

parser = argparse.ArgumentParser(description='Pareto Invariant Risk Minimization')
# General
parser.add_argument('--dataset', type=str,
                    help="Name of dataset, choose from amazon, camelyon, "
                         "rxrx, civil, fmow, iwildcam, poverty")
parser.add_argument('--algorithm', type=str,
                    help='training scheme, choose between fish or erm.')
parser.add_argument('--experiment', type=str, default='.',
                    help='experiment name, set as . for automatic naming.')
parser.add_argument('--data_dir', type=str,
                    help='path to data dir')
parser.add_argument('--exp_dir', type=str, default="",
                    help='path to save results of experiments')
parser.add_argument('--stratified', action='store_true', default=False,
                    help='whether to use stratified sampling for classes')

parser.add_argument('--sample_domains', type=int, default=-1)
parser.add_argument('--epochs', type=int, default=-1)
parser.add_argument('--batch_size', type=int, default=-1)
parser.add_argument('--print_iters', type=int, default=1)
parser.add_argument('--eval_iters', type=int, default=-1)
parser.add_argument('--lr', type=float, default=-1)
parser.add_argument('--momentum', type=float, default=-1)
parser.add_argument('--penalty_weight','-p', type=float, default=1)
parser.add_argument('--penalty_weight2','-p2', type=float, default=-1)   # if there is another penalty weight to be tuned
parser.add_argument('--eps', type=float, default=1e-4)   # if there is another penalty weight to be tuned
parser.add_argument('--preference_choice','-pc',type=int,default=0)
parser.add_argument('--num_workers','-nw',type=int,default=4)
parser.add_argument('--frozen', action='store_true', default=False) # whether to frozen the featurizer
parser.add_argument('--adjust_irm', '-ai',action='store_true', default=False) # whether to adjust some negative irm penalties in pair by adding up a positive number
parser.add_argument('--adjust_loss', '-al',action='store_true', default=False) # whether to adjust some negative irm penalties in pair by multiplying a negative number

parser.add_argument('--need_pretrain', action='store_true')
parser.add_argument('--adjust_lr', '-alr',action='store_true', default=False) # whether to adjust lr as scheduled after pretraining
parser.add_argument('--pretrain_iters', type=int,default=-1)
parser.add_argument('--use_old', action='store_true')
parser.add_argument('--no_plot', action='store_true')
parser.add_argument('--no_test', action='store_true')
parser.add_argument('--no_sch', action='store_true')    # not to use any scheduler
parser.add_argument('--opt', type=str, default='')
parser.add_argument('--exp_name', type=str, default='')
parser.add_argument('--scheduler', type=str, default='')
# Computation
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA use')
parser.add_argument('--seed', type=int, default=-1,
                    help='random seed, set as -1 for random.')
parser.add_argument('--no_wandb', action='store_true', default=False) # whether not to use wandb
parser.add_argument('--no_drop_last', action='store_true', default=True) # whether not to drop last batch


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
# overwrite some arguments
batch_size = args.batch_size
print_iters = args.print_iters
pretrain_iters = args.pretrain_iters
epochs = args.epochs
optimiser = args.opt
args_dict = args.__dict__
args_dict.update(dataset_defaults[args.dataset])
args = argparse.Namespace(**args_dict)

if len(args.exp_dir) == 0:
    args.exp_dir = args.data_dir
os.environ["WANDB_DIR"] = args.exp_dir

# experiment directory setup
args.experiment = f"{args.dataset}_{args.algorithm}" \
    if args.experiment == '.' else args.experiment
directory_name = os.path.join(args.exp_dir,'experiments/{}'.format(args.experiment))
if not os.path.exists(directory_name):
    os.makedirs(directory_name)
runPath = mkdtemp(prefix=runId, dir=directory_name)

    
if batch_size>0:
    args.batch_size=batch_size
if print_iters>0:
    args.print_iters = print_iters
if pretrain_iters>0:
    args.pretrain_iters = pretrain_iters
if len(optimiser)>0:
    args.optimiser = optimiser
    


exp_name = f"{args.experiment}"
if len(args.exp_name)>0:
    args.exp_name ="_"+args.exp_name

if args.algorithm.lower() in ['pair']:
    exp_name += f"_pc{args.preference_choice}"
else:
    exp_name += f"_p{args.penalty_weight}"
    
if args.sample_domains>0:
    exp_name += f"_meta{args.sample_domains}"
    args.meta_steps = args.sample_domains
if args.frozen:
    exp_name += "_frozen"
if epochs>0:
    exp_name += f"_ep{epochs}"
    args.epochs = epochs
exp_name += f"_{args.optimiser}_lr{args.lr}{args.exp_name}_seed{args.seed}"

os.environ["WANDB_NAME"]=exp_name.replace("_","/")
group_name = "/".join(exp_name.split("_")[:-1]) # seed don't participate in the grouping

if args.dataset.lower() in ["poverty"]:
    dataset_name_wandb = args.dataset+"_avg"
else:
    dataset_name_wandb = args.dataset
if not args.no_wandb:
    # raise Exception("Please specify your own parameters if you wish to use wandb")
    wandb_run = wandb.init(project=dataset_name_wandb, entity="entity_name",group=group_name,id=wandb.util.generate_id())
    wandb.config = args


# Choosing and saving a random seed for reproducibility
if args.seed == -1:
    args.seed = int(torch.randint(0, 2 ** 32 - 1, (1,)).item())
set_seed(args.seed)

# logging setup
sys.stdout = Logger('{}/run.log'.format(runPath))
print('RunID:' + runPath)
with open('{}/args.json'.format(runPath), 'w') as fp:
    json.dump(args.__dict__, fp)
torch.save(args, '{}/args.rar'.format(runPath))

# load model
modelC = getattr(models, args.dataset)
train_loader, tv_loaders,dataset = modelC.getDataLoaders(args, device=device)
val_loader, test_loader = tv_loaders['val'], tv_loaders['test']
model = modelC(args, weights=None).to(device)

# assert args.optimiser in ['SGD', 'Adam'], "Invalid choice of optimiser, choose between 'Adam' and 'SGD'"
if args.optimiser.lower() in ['sgd','adam']:
    opt = getattr(optim, args.optimiser)
else:
    raise Exception("Invalid choice of optimiser")
if args.lr>0:
    args.optimiser_args['lr'] = args.lr
# pop up unnecessary configs
if args.optimiser.lower() not in ['adam'] and 'amsgrad' in args.optimiser_args.keys():
    args.optimiser_args.pop('amsgrad')
if args.momentum > 0:
    args.optimiser_args['momentum'] = args.momentum

if args.dataset.lower() in ["poverty"]:
    classifier = model.enc.fc
elif args.dataset.lower() in ["iwildcam","rxrx"]:
    classifier = model.fc
else:
    classifier = model.classifier
trainable_params = classifier.parameters() if args.frozen else model.parameters()
optimiserC = opt(trainable_params, **args.optimiser_args)
predict_fn, criterion = return_predict_fn(args.dataset), return_criterion(args.dataset)



if args.algorithm not in ['erm'] and not args.adjust_lr:
    n_train_steps = train_loader.dataset.training_steps*args.epochs
else:
    n_train_steps = len(train_loader) * args.epochs 
n_train_steps += (args.need_pretrain and args.pretrain_iters>0 and not args.use_old)*args.pretrain_iters

if args.no_sch:
    args.scheduler = None

if args.scheduler is not None and len(args.scheduler)>0:
    scheduler = initialize_scheduler(args, optimiserC, n_train_steps)
else:
    scheduler = None

if args.adjust_lr:
    print("Adjusting learning rate as scheduled after pretraining...")
    n_iters = 0
    pretrain_iters = args.pretrain_iters
    pretrain_epochs = int(np.ceil(pretrain_iters/len(train_loader)))
    pbar = tqdm.tqdm(total = pretrain_iters)
    for epoch in range(pretrain_epochs):
        for i in range(len(train_loader)):
            if scheduler is not None and scheduler.step_every_batch:
                scheduler.step()
            # display progress
            pbar.set_description(f"Pretrain {n_iters}/{pretrain_iters} iters")
            pbar.update(1)
        if scheduler is not None and not scheduler.step_every_batch:
            scheduler.step()
elif args.need_pretrain and args.pretrain_iters>0 and args.use_old:
    if args.scheduler is not None and len(args.scheduler)>0:
        try:
            if 'num_warmup_steps' in  args.scheduler_kwargs.keys():
                args.scheduler_kwargs['num_warmup_steps'] = 0
        except Exception as e:
            print(e)
        scheduler = initialize_scheduler(args, optimiserC, n_train_steps)
    else:
        scheduler = None
print(optimiserC,scheduler)

def pretrain(train_loader, pretrain_iters, save_path=None):
    aggP = defaultdict(list)
    aggP['val_stat'] = [0.]

    n_iters = 0
    pretrain_epochs = int(np.ceil(pretrain_iters/len(train_loader)))
    pbar = tqdm.tqdm(total = pretrain_iters)
    for epoch in range(pretrain_epochs):
        for i, data in enumerate(train_loader):
            model.train()
            # get the inputs
            x, y = unpack_data(data, device)
            optimiserC.zero_grad()
            y_hat = model(x,frozen_mode=args.frozen)
            loss = criterion(y_hat, y)
            loss.backward()
            optimiserC.step()
            if scheduler is not None and scheduler.step_every_batch:
                scheduler.step()
            n_iters += 1
            # display progress
            pbar.set_description(f"Pretrain {n_iters}/{pretrain_iters} iters")
            pbar.update(1)
            if (i + 1) % args.eval_iters == 0 and args.eval_iters != -1:
                test(val_loader, aggP, loader_type='val', verbose=False)
                test(test_loader, aggP, loader_type='test', verbose=False)
                if save_path is None:
                    save_path = runPath
                save_best_model(model, save_path, aggP, args)

            if n_iters == pretrain_iters:
                print("Pretrain is done!")
                test(val_loader, aggP, loader_type='val', verbose=False)
                test(test_loader, aggP, loader_type='test', verbose=False)
                if save_path is None:
                    save_path = runPath
                # save the model at last pretrain epoch no matter whatever
                save_best_model(model, save_path, aggP, args, pretrain=True)
                break
        if scheduler is not None and not scheduler.step_every_batch:
            scheduler.step()
    pbar.close()

    model.load_state_dict(torch.load(save_path + '/model.rar'))
    print('Finished ERM pre-training!')

def train_erm(train_loader, epoch, agg):
    running_loss = 0
    total_iters = len(train_loader)
    print('\n====> Epoch: {:03d} '.format(epoch))
    for i, data in enumerate(train_loader):
        model.train()
        # get the inputs
        x, y = unpack_data(data, device)
        optimiserC.zero_grad()
        y_hat = model(x,frozen_mode=args.frozen)
        loss = criterion(y_hat, y)
        loss.backward()
        optimiserC.step()
        if scheduler is not None and scheduler.step_every_batch:
            scheduler.step()
        running_loss += loss.item()
        # print statistics
        if (i + 1) % args.print_iters == 0 and args.print_iters != -1 and args.algorithm != 'fish':
            if not args.no_wandb:
                wandb.log({ "loss": loss.item()})
            agg['train_loss'].append(running_loss / args.print_iters)
            agg['losses'].append([running_loss / args.print_iters])
            agg['train_iters'].append(i+1+epoch*total_iters)
            print('iteration {:05d}/{:d}: loss: {:6.3f}'.format(i + 1, total_iters, running_loss / args.print_iters))
            if i % args.eval_iters == 0 and args.eval_iters != -1:    
                test(val_loader, agg, loader_type='val')
                test(test_loader, agg, loader_type='test')
                if not args.no_wandb:
                    wandb.log({"val_acc":agg['val_stat'][-1]})
                    wandb.log({"test_acc":agg['test_stat'][-1]})
                running_loss=0
                model.train()
                save_best_model(model, runPath, agg, args)



from wilds.common.utils import split_into_groups
import torch.autograd as autograd
import torch.nn.functional as F
scale = torch.tensor(1.).to(device).requires_grad_()
def irm_penalty(losses, pos=-1, adjust=False):
    grad_1 = autograd.grad(losses[0::2].mean(), [scale], create_graph=True)[0]
    grad_2 = autograd.grad(losses[1::2].mean(), [scale], create_graph=True)[0]
    result = torch.sum(grad_1 * grad_2)
    if pos>0 and not adjust:
        # grad = autograd.grad(losses.mean(), [scale], create_graph=True)[0]
        # result = torch.sum(grad.pow(2))
        result += pos
    if result<0 and adjust:
        grad = autograd.grad(losses.mean(), [scale], create_graph=True)[0]
        result = torch.sum(grad.pow(2))
    return result

def train_irmx(train_loader, epoch, agg):
    model.train()
    train_loader.dataset.reset_batch()
    i = 0
    print('\n====> Epoch: {:03d} '.format(epoch))
    running_loss = 0
    total_iters = len(train_loader)
    running_losses = []
    while sum([l > 1 for l in train_loader.dataset.batches_left.values()]) >= args.meta_steps:
        model.train()
        i += 1
        # sample `meta_steps` number of domains to use for the inner loop
        domains = sample_domains(train_loader, args.meta_steps, args.stratified).tolist()
        # print(domains)
        avg_loss = 0.
        penalty = 0.
        # overall_losses = F.cross_entropy(scale * results['y_pred'],results['y_true'],reduction="none")
        losses_bygroup = []

        # inner loop update
        for domain in domains:
            data = train_loader.dataset.get_batch(domain)
            x, y = unpack_data(data, device)
            y_hat = model(x,frozen_mode=args.frozen)
            # loss = criterion(y_hat, y)
            if 'poverty'in args.dataset.lower():
                loss = F.mse_loss(scale*y_hat,y,reduction="none")
            else:
                loss = F.cross_entropy(scale * y_hat,y,reduction="none")
            losses_bygroup.append(loss.mean())
            penalty += irm_penalty(loss)
            avg_loss += loss.mean()
        avg_loss /= args.meta_steps
        penalty /= args.meta_steps
        # losses = losses_bygroup+[ penalty, torch.stack(losses_bygroup).var()]
        losses = [avg_loss, penalty, torch.stack(losses_bygroup).var()]
        # agg['losses'].append([l.item() for l in losses])
        if len(running_losses)==0:
            running_losses = [0]*len(losses)
        for (j,loss) in enumerate(running_losses):
            running_losses[j]+=losses[j].item()
        # print([l.item() for l in losses],sol)
        optimiserC.zero_grad()
        # loss = scales.dot(torch.stack(losses))
        if args.penalty_weight2 > 0:
            loss = avg_loss+args.penalty_weight*penalty+args.penalty_weight2*torch.stack(losses_bygroup).var()
        else:
            loss = avg_loss+args.penalty_weight*(penalty+torch.stack(losses_bygroup).var())
        # print(loss)
        loss.backward()
        optimiserC.step()
        if scheduler is not None and scheduler.step_every_batch:
            scheduler.step()
        running_loss += loss.item()

        # log the number of batches left for each domain
        for domain in domains:
            train_loader.dataset.batches_left[domain] = \
                train_loader.dataset.batches_left[domain] - 1 \
                if train_loader.dataset.batches_left[domain] > 1 else 1

        if i % args.print_iters == 0 and args.print_iters != -1:            
            print(avg_loss,penalty)
            agg['losses'].append([l / args.print_iters for l in running_losses])
            if not args.no_wandb:
                wandb.log({ "loss": loss.item(),
                            "erm_loss": agg['losses'][-1][0],
                            "irm_loss": agg['losses'][-1][1],
                            "vrex_loss": agg['losses'][-1][2],
                            })
            running_losses = [0]*len(losses)
            # agg['losses'].append([l.item() for l in losses])
            agg['train_loss'].append(running_loss / args.print_iters)
            agg['train_iters'].append(i+1+epoch*total_iters)
            print('iteration {:05d}/{:d}: loss: {:6.3f}'.format(i + 1, total_iters, running_loss / args.print_iters))
            if i % args.eval_iters == 0 and args.eval_iters != -1:    
                test(val_loader, agg, loader_type='val')
                test(test_loader, agg, loader_type='test')
                if not args.no_wandb:
                    wandb.log({"val_acc":agg['val_stat'][-1]})
                    wandb.log({"test_acc":agg['test_stat'][-1]})
                model.train()
                save_best_model(model, runPath, agg, args)

def train_irm(train_loader, epoch, agg):
    model.train()
    train_loader.dataset.reset_batch()
    i = 0
    print('\n====> Epoch: {:03d} '.format(epoch))
    running_loss = 0
    total_iters = len(train_loader)
    running_losses = []
    while sum([l > 1 for l in train_loader.dataset.batches_left.values()]) >= args.meta_steps:
        model.train()
        i += 1
        # sample `meta_steps` number of domains to use for the inner loop
        domains = sample_domains(train_loader, args.meta_steps, args.stratified).tolist()
        # print(domains)
        avg_loss = 0.
        penalty = 0.
        # overall_losses = F.cross_entropy(scale * results['y_pred'],results['y_true'],reduction="none")
        losses_bygroup = []

        # inner loop update
        for domain in domains:
            data = train_loader.dataset.get_batch(domain)
            x, y = unpack_data(data, device)
            y_hat = model(x,frozen_mode=args.frozen)
            # loss = criterion(y_hat, y)
            if 'poverty'in args.dataset.lower():
                loss = F.mse_loss(scale*y_hat,y,reduction="none")
            else:
                loss = F.cross_entropy(scale * y_hat,y,reduction="none")
            losses_bygroup.append(loss.mean())
            penalty += irm_penalty(loss)
            avg_loss += loss.mean()
        avg_loss /= args.meta_steps
        penalty /= args.meta_steps
        # losses = losses_bygroup+[ penalty, torch.stack(losses_bygroup).var()]
        losses = [avg_loss, penalty, torch.stack(losses_bygroup).var()]
        # agg['losses'].append([l.item() for l in losses])
        if len(running_losses)==0:
            running_losses = [0]*len(losses)
        for (j,loss) in enumerate(running_losses):
            running_losses[j]+=losses[j].item()
        # print([l.item() for l in losses],sol)
        optimiserC.zero_grad()
        
        # loss = scales.dot(torch.stack(losses))
        loss = avg_loss+args.penalty_weight*penalty
        # print(loss)
        loss.backward()
        optimiserC.step()
        running_loss += loss.item()
        if scheduler is not None and scheduler.step_every_batch:
            scheduler.step()
        # log the number of batches left for each domain
        for domain in domains:
            train_loader.dataset.batches_left[domain] = \
                train_loader.dataset.batches_left[domain] - 1 \
                if train_loader.dataset.batches_left[domain] > 1 else 1

        if i % args.print_iters == 0 and args.print_iters != -1:            
            print(avg_loss,penalty)
            agg['losses'].append([l / args.print_iters for l in running_losses])
            if not args.no_wandb:
                wandb.log({ "loss": loss.item(),
                            "erm_loss": agg['losses'][-1][0],
                            "irm_loss": agg['losses'][-1][1],
                            "vrex_loss": agg['losses'][-1][2],
                            })
            running_losses = [0]*len(losses)
            # agg['losses'].append([l.item() for l in losses])
            agg['train_loss'].append(running_loss / args.print_iters)
            agg['train_iters'].append(i+1+epoch*total_iters)
            print('iteration {:05d}/{:d}: loss: {:6.3f}'.format(i + 1, total_iters, running_loss / args.print_iters))
            if i % args.eval_iters == 0 and args.eval_iters != -1:    
                test(val_loader, agg, loader_type='val')
                test(test_loader, agg, loader_type='test')
                if not args.no_wandb:
                    wandb.log({"val_acc":agg['val_stat'][-1]})
                    wandb.log({"test_acc":agg['test_stat'][-1]})
                model.train()
                save_best_model(model, runPath, agg, args)

def train_vrex(train_loader, epoch, agg):
    model.train()
    train_loader.dataset.reset_batch()
    i = 0
    print('\n====> Epoch: {:03d} '.format(epoch))
    running_loss = 0
    total_iters = len(train_loader)
    running_losses = []
    while sum([l > 1 for l in train_loader.dataset.batches_left.values()]) >= args.meta_steps:
        model.train()
        i += 1
        # sample `meta_steps` number of domains to use for the inner loop
        domains = sample_domains(train_loader, args.meta_steps, args.stratified).tolist()
        # print(domains)
        avg_loss = 0.
        penalty = 0.
        # overall_losses = F.cross_entropy(scale * results['y_pred'],results['y_true'],reduction="none")
        losses_bygroup = []

        # inner loop update
        for domain in domains:
            data = train_loader.dataset.get_batch(domain)
            x, y = unpack_data(data, device)
            y_hat = model(x,frozen_mode=args.frozen)
            # loss = criterion(y_hat, y)
            if 'poverty'in args.dataset.lower():
                loss = F.mse_loss(scale*y_hat,y,reduction="none")
            else:
                loss = F.cross_entropy(scale * y_hat,y,reduction="none")
            losses_bygroup.append(loss.mean())

            penalty += irm_penalty(loss)
            avg_loss += loss.mean()
        avg_loss /= args.meta_steps
        penalty /= args.meta_steps
        losses = [avg_loss, penalty, torch.stack(losses_bygroup).var()]
        if len(running_losses)==0:
            running_losses = [0]*len(losses)
        for (j,loss) in enumerate(running_losses):
            running_losses[j]+=losses[j].item()

        # print([l.item() for l in losses],sol)
        optimiserC.zero_grad()
        # loss = scales.dot(torch.stack(losses))
        loss = avg_loss+args.penalty_weight*torch.stack(losses_bygroup).var()
        # print(loss)
        loss.backward()
        optimiserC.step()
        if scheduler is not None and scheduler.step_every_batch:
            scheduler.step()
        running_loss += loss.item()

        # log the number of batches left for each domain
        for domain in domains:
            train_loader.dataset.batches_left[domain] = \
                train_loader.dataset.batches_left[domain] - 1 \
                if train_loader.dataset.batches_left[domain] > 1 else 1
        # print(i)
        if i % args.print_iters == 0 and args.print_iters != -1:
            print(avg_loss,penalty)
            agg['losses'].append([l / args.print_iters for l in running_losses])
            if not args.no_wandb:
                wandb.log({ "loss": loss.item(),
                "erm_loss": agg['losses'][-1][0],
                "irm_loss": agg['losses'][-1][1],
                "vrex_loss": agg['losses'][-1][2],
                })
            running_losses = [0]*len(losses)
            # agg['losses'].append([l.item() for l in losses])
            agg['train_loss'].append(running_loss / args.print_iters)
            agg['train_iters'].append(i+1+epoch*total_iters)
            print('iteration {:05d}/{:d}: loss: {:6.3f}'.format(i + 1, total_iters, running_loss / args.print_iters))
            if i % args.eval_iters == 0 and args.eval_iters != -1:    
                test(val_loader, agg, loader_type='val')
                test(test_loader, agg, loader_type='test')
                if not args.no_wandb:
                    wandb.log({"val_acc":agg['val_stat'][-1]})
                    wandb.log({"test_acc":agg['test_stat'][-1]})
                model.train()
                save_best_model(model, runPath, agg, args)


amplify = 1e2 if (not args.adjust_irm and not args.adjust_loss) else 1
preference = get_preference(args.preference_choice)
n_tasks = 1+2
preference[1]/=amplify+1e-6
pair_optimizer = PAIR(trainable_params,optimiserC,preference=preference,eps=args.eps)
descent = 0
def train_pair(train_loader, epoch, agg):
    model.train()
    train_loader.dataset.reset_batch()
    i = 0
    print('\n====> Epoch: {:03d} '.format(epoch))
    running_loss = 0
    total_iters = len(train_loader)
    running_losses = []
    while sum([l > 1 for l in train_loader.dataset.batches_left.values()]) >= args.meta_steps:
        model.train()
        i += 1
        # sample `meta_steps` number of domains to use for the inner loop
        domains = sample_domains(train_loader, args.meta_steps, args.stratified).tolist()

        avg_loss = 0.
        penalty = 0.
        # overall_losses = F.cross_entropy(scale * results['y_pred'],results['y_true'],reduction="none")
        losses_bygroup = []
        y_hats = []
        # inner loop update
        for domain in domains:
            data = train_loader.dataset.get_batch(domain)
            x, y = unpack_data(data, device)
            y_hat = model(x,frozen_mode=args.frozen)
            if 'poverty'in args.dataset.lower():
                loss = F.mse_loss(scale*y_hat,y,reduction="none")
            else:
                loss = F.cross_entropy(scale * y_hat,y,reduction="none")
            losses_bygroup.append(loss.mean())

            if args.adjust_loss:
                penalty += irm_penalty(loss)
            else:
                penalty += irm_penalty(loss,pos=amplify,adjust=args.adjust_irm)
            avg_loss += loss.mean()
        avg_loss /= args.meta_steps
        penalty /= args.meta_steps
        losses = [avg_loss, penalty, torch.stack(losses_bygroup).var()]
        if len(running_losses)==0:
            running_losses = [0]*len(losses)
        for (j,loss) in enumerate(running_losses):
            running_losses[j]+=losses[j].item()
        pair_optimizer.zero_grad()
        pair_optimizer.set_losses(losses)
        pair_loss, moo_losses, mu_rl, alphas = pair_optimizer.step()
        
        if scheduler is not None and scheduler.step_every_batch:
            scheduler.step()
        running_loss += pair_loss

        # log the number of batches left for each domain
        for domain in domains:
            train_loader.dataset.batches_left[domain] = \
                train_loader.dataset.batches_left[domain] - 1 \
                if train_loader.dataset.batches_left[domain] > 1 else 1

        if i % args.print_iters == 0 and args.print_iters != -1:            
            agg['losses'].append([l / args.print_iters for l in running_losses])
            # compensate the irm penalty
            if not args.adjust_irm  and not args.adjust_loss:
                agg['losses'][-1][1] -= amplify

            if not args.no_wandb:
                wandb.log({ "loss": loss.item(),
                "erm_loss": agg['losses'][-1][0],
                "irm_loss": agg['losses'][-1][1],
                "vrex_loss": agg['losses'][-1][2],
                "mu_rl":mu_rl,
                "erm_alpha":alphas[0],
                "irm_alpha":alphas[1],
                "vrex_alpha":alphas[2]
                })
            running_losses = [0]*len(losses)
            # agg['losses'].append([l.item() for l in losses])
            agg['train_loss'].append(running_loss / args.print_iters)
            agg['train_iters'].append(i+1+epoch*total_iters)
            print('iteration {:05d}/{:d}: loss: {:6.3f}'.format(i + 1, total_iters, running_loss / args.print_iters))
            if i % args.eval_iters == 0 and args.eval_iters != -1:    
                test(val_loader, agg, loader_type='val')
                test(test_loader, agg, loader_type='test')
                if not args.no_wandb:
                    wandb.log({"val_acc":agg['val_stat'][-1]})
                    wandb.log({"test_acc":agg['test_stat'][-1]})
                model.train()
                save_best_model(model, runPath, agg, args)


def train_fish(train_loader, epoch, agg):
    model.train()
    train_loader.dataset.reset_batch()
    i = 0
    print('\n====> Epoch: {:03d} '.format(epoch))
    opt_inner_pre = None
    running_losses = []
    
    while sum([l > 1 for l in train_loader.dataset.batches_left.values()]) >= args.meta_steps:
        i += 1
        # sample `meta_steps` number of domains to use for the inner loop
        domains = sample_domains(train_loader, args.meta_steps, args.stratified).tolist()
        

        # prepare model for inner loop update
        model_inner = copy.deepcopy(model)
        
        model_inner.train()
        if args.dataset.lower() in ["poverty"]:
            classifier = model_inner.enc.fc
        elif args.dataset.lower() in ["iwildcam","rxrx"]:
            classifier = model_inner.fc
        else:
            classifier = model_inner.classifier
        inner_trainable_params = classifier.parameters() if args.frozen else model_inner.parameters()
        opt_inner = opt(inner_trainable_params, **args.optimiser_args)
        if opt_inner_pre is not None and args.reload_inner_optim:
            opt_inner.load_state_dict(opt_inner_pre)

        penalty = 0.
        avg_loss = 0
        losses_bygroup = []
        # inner loop update
        for domain in domains:
            data = train_loader.dataset.get_batch(domain)
            x, y = unpack_data(data, device)
            opt_inner.zero_grad()
            y_hat = model_inner(x)
            loss = criterion(y_hat, y)
            loss.backward()
            opt_inner.step()
            losses_bygroup.append(loss.mean())
            if 'poverty'in args.dataset.lower():
                cur_loss = F.mse_loss(scale*y_hat,y,reduction="none")
            else:
                cur_loss = F.cross_entropy(scale * y_hat,y,reduction="none")
            penalty += irm_penalty(cur_loss)
            avg_loss += loss.mean()
        
        avg_loss /= args.meta_steps
        penalty /= args.meta_steps
        # losses = losses_bygroup+[ penalty, torch.stack(losses_bygroup).var()]
        losses = [avg_loss, penalty, torch.stack(losses_bygroup).var()]
        # agg['losses'].append([l.item() for l in losses])
        if len(running_losses)==0:
            running_losses = [0]*len(losses)
        for (j,loss) in enumerate(running_losses):
            running_losses[j]+=losses[j].item()

        opt_inner_pre = opt_inner.state_dict()
        # fish update
        meta_weights = fish_step(meta_weights=model.state_dict(),
                                 inner_weights=model_inner.state_dict(),
                                 meta_lr=args.meta_lr / args.meta_steps)
        model.reset_weights(meta_weights)
        # log the number of batches left for each domain
        for domain in domains:
            train_loader.dataset.batches_left[domain] = \
                train_loader.dataset.batches_left[domain] - 1 \
                if train_loader.dataset.batches_left[domain] > 1 else 1

        if (i + 1) % args.print_iters == 0 and args.print_iters != -1:
            agg['losses'].append([l / args.print_iters for l in running_losses])
            if not args.no_wandb:
                wandb.log({ "loss": avg_loss,
                "erm_loss": agg['losses'][-1][0],
                "irm_loss": agg['losses'][-1][1],
                "vrex_loss": agg['losses'][-1][2],
                })
            print(f"iteration {(i + 1):05d}: {agg['losses'][-1]}")
            running_losses = [0]*len(losses)
            if i % args.eval_iters == 0 and args.eval_iters != -1:    
                test(val_loader, agg, loader_type='val')
                test(test_loader, agg, loader_type='test')
                if not args.no_wandb:
                    wandb.log({"val_acc":agg['val_stat'][-1]})
                    wandb.log({"test_acc":agg['test_stat'][-1]})
                model.train()
                save_best_model(model, runPath, agg, args)

def test(test_loader, agg, loader_type='test', verbose=True, save_ypred=False, return_last=False):
    model.eval()
    yhats, ys, metas = [], [], []
    import timeit
    with torch.no_grad():
        a = timeit.default_timer()
        for i, (x, y, meta) in enumerate(test_loader):
            # get the inputs
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            ys.append(y)
            yhats.append(y_hat)
            metas.append(meta)
            # print(timeit.default_timer()-a)
            # a = timeit.default_timer()
        ypreds, ys, metas = predict_fn(torch.cat(yhats)), torch.cat(ys), torch.cat(metas)
        if save_ypred:
            if args.dataset == 'poverty':
                save_name = f"{args.dataset}_split:{loader_type}_fold:" \
                            f"{['A', 'B', 'C', 'D', 'E'][args.seed]}" \
                            f"_epoch:best_pred.csv"
            else:
                save_name = f"{args.dataset}_split:{loader_type}_seed:" \
                            f"{args.seed}_epoch:best_pred.csv"
            with open(f"{runPath}/{save_name}", 'w') as f:
                writer = csv.writer(f)
                writer.writerows(ypreds.unsqueeze(1).cpu().tolist())
        test_val = test_loader.dataset.eval(ypreds.cpu(), ys.cpu(), metas)
        agg[f'{loader_type}_stat'].append(test_val[0][args.selection_metric])
        if verbose:
            print(f"=============== {loader_type} ===============\n{test_val[-1]}")
    if return_last:
        return test_val[0][args.selection_metric]


if __name__ == '__main__':
    try:
        if args.need_pretrain and args.pretrain_iters != 0:
            pretrain_path = os.path.join(args.exp_dir,"experiments",args.dataset,str(args.seed))
            if not os.path.exists(pretrain_path):
                os.makedirs(pretrain_path)
            if args.use_old:
                model.load_state_dict(torch.load(pretrain_path + f'/model.rar'))
                print(f"Load pretrained model from {pretrain_path}")
            else:
                print("="*30 + "ERM pretrain" + "="*30)
                pretrain(train_loader, args.pretrain_iters, save_path=pretrain_path)

        torch.cuda.empty_cache()
        print("="*30 + f"Training: {args.algorithm}" + "="*30)
        train = locals()[f'train_{args.algorithm}']
        agg = defaultdict(list)
        agg['val_stat'] = [0.]

        for epoch in range(args.epochs):
            train(train_loader, epoch, agg)

            test(val_loader, agg, loader_type='val')
            test(test_loader, agg, loader_type='test')
            save_best_model(model, runPath, agg, args)
            if not args.no_wandb:
                wandb.log({"val_acc":agg['val_stat'][-1]})
                wandb.log({"test_acc":agg['test_stat'][-1]})
            if scheduler is not None and not scheduler.step_every_batch:
                scheduler.step()
                print(optimiserC)

        model.load_state_dict(torch.load(runPath + '/model.rar'))
        print('Finished training! Loading best model...')
        test_acc = 0
        for split, loader in tv_loaders.items():
            tmp_acc = test(loader, agg, loader_type=split, save_ypred=True,return_last=True)
            if split=="test":
                test_acc = tmp_acc

        import matplotlib.pyplot as plt
        if not args.no_plot:
            folder_name = os.path.join(args.exp_dir,"plots",f"{args.dataset}_{args.algorithm}")
            if not os.path.exists(folder_name):
                os.mkdir(folder_name)
            num_epochs = len(agg['losses'])
            plt.title(exp_name)
            fig, ax1 = plt.subplots()
            ax1.set_xlabel("epoch")
            ax1.set_ylabel("test acc")
            
            
            if args.algorithm in ['pair','irm','vrex','fish',"irmx"]:
                ax2 = ax1.twinx()
                ax2.set_ylabel("penalty")
                if len(agg['losses'][0])>=3:
                    irm_pens = np.array([log_i[-2] for log_i in agg['losses']])
                    vrex_pens = np.array([log_i[-1] for log_i in agg['losses']])
                    ax2.plot(np.arange(num_epochs),irm_pens,label=f'irm_pen',c='r',alpha=0.2)
                    ax2.plot(np.arange(num_epochs),vrex_pens,label=f'vrex_pen',c='g',alpha=0.2)
                    erm_pens = np.array([log_i[-3] for log_i in agg['losses']])
                else:
                    irm_pens = np.array([log_i[-1] for log_i in agg['losses']])
                    ax2.plot(np.arange(num_epochs),irm_pens,label=f'{args.algorithm}_pen',c='r',alpha=0.2)
                    erm_pens = np.array([log_i[-2] for log_i in agg['losses']])
            
                max_ratio = irm_pens.max()
                ax2.set_title(f"{exp_name}: {test_acc}")
                # control the scale of erm loss w.r.t. others
                erm_pens = np.clip(erm_pens,erm_pens.min(),erm_pens.min()*max_ratio)
                ax1.plot(np.arange(num_epochs),erm_pens,label=f'erm_pen')
                # ask matplotlib for the plotted objects and their labels
                lines, labels = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax2.legend(lines + lines2, labels + labels2, loc=0)
            else:
                erm_pens = np.array([log_i[0] for log_i in agg['losses']])
                ax1.plot(np.arange(num_epochs),erm_pens,label=f'erm_pen')
            
            plt.savefig(os.path.join(folder_name,f"{exp_name}.png"))
            plt.close()
        torch.save(agg,os.path.join(folder_name,f"{exp_name}_agg.pt"))
        if not args.no_wandb:
            wandb.finish()
    except Exception as e:
        traceback.print_exc()
        print(e)
        if not args.no_wandb:
            wandb.finish(-1)
            print("Exceptions found, delete all wandb files")
            import shutil
            shutil.rmtree(wandb_run.dir.replace("/files",""))
