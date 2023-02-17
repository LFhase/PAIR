import numpy as np 
from torch import nn
import torch
import torch.nn.functional as F

import numpy as np

def parse_bool(v):
    if v.lower()=='true':
        return True
    elif v.lower()=='false':
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def pretty_print(*values):
    col_width = 13
    def format_val(v):
        if not isinstance(v, str):
            v = np.array2string(v, precision=5, floatmode='fixed')
        return v.ljust(col_width)
    str_values = [format_val(v) for v in values]
    print(" ".join(str_values))

# Define loss function helpers
def mean_weight(weights):
    
    weight = copy.deepcopy(weights[0])
    for key in weight:
        for val in weights[1:]:
            weight[key] += val[key]

    for key in weight:
        weight[key] /= len(weights)

    return weight


def mean_nll(logits, y, reduction='mean'):
    return nn.functional.binary_cross_entropy_with_logits(logits, y,reduction=reduction)

def mean_mse(logits, y, reduction = 'mean'):
    if reduction == 'mean':
        return ((logits - (2*y-1))**2).mean()/2
    elif reduction == 'none':
        return ((logits - (2*y-1))**2)/2

def mean_accuracy(logits, y, reduction = 'mean'):
    if logits.size(1) == 1:
        preds = (logits > 0.).float()
        if reduction == 'mean':
            return ((preds - y).abs() < 1e-2).float().mean()
        else:
            return ((preds - y).abs() < 1e-2).float()
    else:
        if reduction == 'mean':
            return (logits.argmax(1).eq(y).float()).mean()
        else:
            return (logits.argmax(1).eq(y).float())

def correct_pred(logits, y):
    if logits.size(1) == 1:
        preds = (logits > 0.).float()
        correct = ((preds - y).abs() < 1e-2).float().cpu().detach().numpy().flatten().astype(bool)

    else:
        correct = logits.argmax(1).eq(y).float().cpu().detach().numpy().flatten().astype(bool)
   
    return correct, ~correct 

def validation(topmlp, mlp, envs, test_envs, lossf):
        
    with torch.no_grad():
        for env in envs + test_envs:
            logits = topmlp(mlp(env['images']))

            env['nll'] = lossf(logits, env['labels'])
            env['acc'] = mean_accuracy(logits, env['labels'])

    test_worst_loss = torch.stack([env['nll'] for env in test_envs]).max()
    test_worst_acc  = torch.stack([env['acc'] for env in test_envs]).min()
    train_loss = torch.stack([env['nll'] for env in envs]).mean()
    train_acc  = torch.stack([env['acc'] for env in envs]).mean()
    
    return train_loss.detach().cpu().numpy(), train_acc.detach().cpu().numpy(), \
    test_worst_loss.detach().cpu().numpy(),test_worst_acc.detach().cpu().numpy()

def validation_details(topmlp, mlp, envs, test_envs, lossf):
        
    with torch.no_grad():
        for env in envs + test_envs:
            logits = topmlp(mlp(env['images']))
            
            env['nll'] = lossf(logits, env['labels'])
            env['acc'] = mean_accuracy(logits, env['labels'])

    train_loss = torch.stack([env['nll'] for env in envs]).mean()
    train_acc  = torch.stack([env['acc'] for env in envs]).mean()
    
    return train_loss.detach().cpu().numpy(), train_acc.detach().cpu().numpy(), \
    [env['nll'].detach().cpu().numpy() for env in test_envs], \
    [env['acc'].detach().cpu().numpy() for env in test_envs]

def validation2(model, envs, test_envs, lossf):
        
    with torch.no_grad():
        for env in envs + test_envs:
            logits = model(env['images'])

            env['nll'] = lossf(logits, env['labels'])
            env['acc'] = mean_accuracy(logits, env['labels'])

    test_worst_loss = torch.stack([env['nll'] for env in test_envs]).max()
    test_worst_acc  = torch.stack([env['acc'] for env in test_envs]).min()
    train_loss = torch.stack([env['nll'] for env in envs]).mean()
    train_acc  = torch.stack([env['acc'] for env in envs]).mean()
    
    return train_loss.detach().cpu().numpy(), train_acc.detach().cpu().numpy(), \
    test_worst_loss.detach().cpu().numpy(),test_worst_acc.detach().cpu().numpy()




# from https://github.com/alinlab/LfF/blob/e66796ec117ea52d2e44176055b7ef7959680a1b/module/loss.py#L8    
class GeneralizedCELoss(nn.Module):

    def __init__(self, q=0.7):
        super(GeneralizedCELoss, self).__init__()
        self.q = q
             
    def forward(self, logits, targets):
        p = F.softmax(logits, dim=1)
        if np.isnan(p.mean().item()):
            raise NameError('GCE_p')
        Yg = torch.gather(p, 1, torch.unsqueeze(targets, 1))
        # modify gradient of cross entropy
        loss_weight = (Yg.squeeze().detach()**self.q)*self.q
        if np.isnan(Yg.mean().item()):
            raise NameError('GCE_Yg')

        loss = F.cross_entropy(logits, targets, reduction='none') * loss_weight

        return loss

# https://github.com/alinlab/LfF/blob/e66796ec117ea52d2e44176055b7ef7959680a1b/util.py#L33
class EMA:
    
    def __init__(self, label, alpha=0.9):
        self.label = label
        self.alpha = alpha
        self.parameter = torch.zeros(label.shape[0])
        self.updated = torch.zeros(label.shape[0])
        
    def update(self, data, index):
        self.parameter[index] = self.alpha * self.parameter[index] + (1-self.alpha*self.updated[index]) * data
        self.updated[index] = 1
        
    def max_loss(self, label):
        label_index = np.where(self.label == label)[0]
        return self.parameter[label_index].max()

