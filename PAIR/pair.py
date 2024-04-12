import copy
import imp
from pickletools import optimize
import torch
from torch.optim.optimizer import Optimizer, required
from torch.autograd import Variable
import traceback
import torch.nn.functional as F
from torch.optim import SGD

class PAIR(Optimizer):
    r"""
    Implements Pareto Invariant Risk Minimization (PAIR) algorithm.
    It is proposed in the ICLR 2023 paper  
    `Pareto Invariant Risk Minimization: Towards Mitigating the Optimization Dilemma in Out-of-Distribution Generalization`
    https://arxiv.org/abs/2206.07766 .

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        optimizer (pytorch optim): inner optimizer
        balancer (str, optional): indicates which MOO solver to use
        preference (list[float], optional): preference of the objectives 
        eps (float, optional): precision up to the preference (default: 1e-04)
        coe (float, optional): L2 regularization weight onto the yielded objective weights (default: 0)
    """

    def __init__(self, params, optimizer=required, balancer="EPO",preference=[1e-8,1-1e-8], eps=1e-4, coe=0, verbose=False):
        # TODO: parameter validty checking
        if eps < 0.0:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        for _pp in preference:
            if  _pp < 0.0:
                raise ValueError("Invalid preference: {}".format(preference))

        self.optimizer = optimizer
        if type(preference) == list:
            preference = np.array(preference)
        self.preference = preference

        self.descent = 0
        self.losses = []
        self.params = params
        if balancer.lower() == "epo":
            self.balancer = EPO(len(self.preference),self.preference,eps=eps,coe=coe,verbose=verbose)
        elif balancer.lower() == "sepo":
            self.balancer = SEPO(len(self.preference),self.preference,eps=eps,coe=coe,verbose=verbose)
        else:
            raise NotImplementedError("Nrot supported balancer")
        defaults = dict(balancer=balancer, preference=self.preference, eps=eps)
        super(PAIR, self).__init__(params, defaults)
    

    def __setstate__(self, state):
        super(PAIR, self).__setstate__(state)
            
    def set_losses(self,losses):
        self.losses = losses

    def step(self, closure=None):
        r"""Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        if len(self.losses) == 0:
            self.optimizer.step()
            alphas = np.zeros(len(self.preference))
            alphas[0] = 1
            return -1, 233, alphas
        else:
            losses = self.losses
            if closure is not None:
                losses = closure()

            pair_loss = 0
            mu_rl = 0
            alphas = 0

            grads = []
            for cur_loss in losses:
                self.optimizer.zero_grad()
                cur_loss.backward(retain_graph=True)
                cur_grad = []
                for group in self.param_groups:
                    for param in group['params']:
                        if param.grad is not None:
                            cur_grad.append(Variable(param.grad.data.clone().flatten(), requires_grad=False))
                grads.append(torch.cat(cur_grad))

            G = torch.stack(grads)
            # if self.get_grad_sim:
            #     grad_sim = get_grad_sim(G,losses,preference=self.preference,is_G=True)
            GG = G @ G.T
            moo_losses = np.stack([l.item() for l in losses])
            reset_optimizer = False
            try:
                # Calculate the alphas from the LP solver
                alpha, mu_rl, reset_optimizer = self.balancer.get_alpha(moo_losses, G=GG.cpu().numpy(), C=True,get_mu=True)
                if self.balancer.last_move == "dom":
                    self.descent += 1
                    print("dom")
            except Exception as e:
                print(traceback.format_exc())
                alpha = None
            if alpha is None:   # A patch for the issue in cvxpy
                alpha = self.preference / np.sum(self.preference)
            
            scales = torch.from_numpy(alpha).float().to(losses[-1].device)
            pair_loss = scales.dot(losses)
            if reset_optimizer:
                self.optimizer.param_groups[0]["lr"]/=5
                # self.optimizer = torch.optim.Adam(self.params,lr=self.optimizer.param_groups[0]["lr"]/5)
            self.optimizer.zero_grad()
            pair_loss.backward()
            self.optimizer.step()
            
            return pair_loss, moo_losses, mu_rl, alpha



import numpy as np
import cvxpy as cp
import cvxopt

class EPO(object):
    r"""
    The original EPO solver proposed in ICML2020
    https://proceedings.mlr.press/v119/mahapatra20a.html
    """
    def __init__(self, m, r, eps=1e-4, coe=0, verbose=False):
        # self.solver = cp.GLPK
        self.solver = cp.GUROBI
        # cvxopt.glpk.options["msg_lev"] = "GLP_MSG_OFF"
        self.m = m
        self.r = r/np.sum(r)
        self.eps = eps
        self.last_move = None
        self.a = cp.Parameter(m)        # Adjustments
        self.C = cp.Parameter((m, m))   # C: Gradient inner products, G^T G
        self.Ca = cp.Parameter(m)       # d_bal^TG
        self.rhs = cp.Parameter(m)      # RHS of constraints for balancing

        self.alpha = cp.Variable(m)     # Variable to optimize
        self.last_alpha = np.zeros_like(r)-1   
        self.coe = coe

        obj_bal = cp.Maximize(self.alpha @ self.Ca)   # objective for balance
        constraints_bal = [self.alpha >= 0, cp.sum(self.alpha) == 1,  # Simplex
                           self.C @ self.alpha >= self.rhs]
        self.prob_bal = cp.Problem(obj_bal, constraints_bal)  # LP balance

        obj_dom = cp.Maximize(cp.sum(self.alpha @ self.C)-self.coe*cp.sum_squares(self.alpha-self.last_alpha))  # obj for descent
        constraints_dom = [self.alpha >= 0, cp.sum(self.alpha) == 1,  # Restrict
                           self.alpha @ self.Ca >= -cp.neg(cp.max(self.Ca)),
                           self.C @ self.alpha >= 0]
        self.prob_dom = cp.Problem(obj_dom, constraints_dom)  # LP dominance


        self.gamma = 0     # Stores the latest Optimum value of the LP problem
        self.mu_rl = 0     # Stores the latest non-uniformity

        self.verbose = verbose
        

    def get_alpha(self, l, G, r=None, C=False, get_mu=False):
        """calculate weights for all objectives given the gradient information

        Args:
            l (ndarray): the values of objective losses
            G (ndarray): inner products of the gradients of each objective loss w.r.t. params
            r (ndarray, optional): adopt this preference if specified
            C (bool, optional): True if the input gradients are inner products
            get_mu (bool, optional): return detailed information if True.

        Returns:
            alpha: the objective weights
            mu_rl (optional): the optimal value to the LP
            reset_optimizer (optional): whether to reset the inner optimizer
        """
        r = self.r if r is None else r
        assert len(l) == len(G) == len(r) == self.m, "length != m"
        l_hat, rl, self.mu_rl, self.a.value = self.adjustments(l, r)
        reset_optimizer = False
        self.C.value = G if C else G @ G.T
        self.Ca.value = self.C.value @ self.a.value

        if self.last_alpha.sum() is None:
            self.last_alpha = np.array(r)
        if self.mu_rl > self.eps:
            J = self.Ca.value > 0
            J_star_idx = np.where(rl == np.max(rl))[0]
            self.rhs.value = self.Ca.value.copy()
            # it's equivalent to setting no constraints to objectives in J
            # as maximize alpha^TCa would trivially satisfy the non-negativity
            self.rhs.value[J] = -np.inf     
            self.rhs.value[J_star_idx] =  0
            
            self.gamma = self.prob_bal.solve(solver=self.solver, verbose=False)
            self.last_move = "bal"
            
            if self.verbose:
                test_alpha = np.ones_like(self.a.value)/self.m
                print(self.last_alpha,self.C.value,self.Ca.value,self.rhs.value)
                print(self.gamma,test_alpha@self.Ca.value, self.alpha.value @ self.C.value)
                print(self.gamma,self.coe*np.linalg.norm(self.alpha.value-self.last_alpha)**2)

        else:
            self.gamma = self.prob_dom.solve(solver=self.solver, verbose=False)
            self.last_move = "dom"
        self.last_alpha = np.array(self.alpha.value)
        
        if get_mu:
            return self.alpha.value, self.mu_rl, reset_optimizer
        
        return self.alpha.value


    def mu(self, rl, normed=False):
        if len(np.where(rl < 0)[0]):
            raise ValueError(f"rl<0 \n rl={rl}")
            return None
        m = len(rl)
        l_hat = rl if normed else rl / rl.sum()
        eps = np.finfo(rl.dtype).eps
        l_hat = l_hat[l_hat > eps]
        return np.sum(l_hat * np.log(l_hat * m))


    def adjustments(self, l, r=1):
        m = len(l)
        rl = r * l
        
        l_hat = rl / rl.sum()
        mu_rl = self.mu(l_hat, normed=True)
        uniformity_div = np.log(l_hat * m) - mu_rl
        div_r = np.array(r)
        a = div_r * uniformity_div

        if self.verbose:
            print(a, rl, div_r, uniformity_div, l_hat, a.dot(l))
        return l_hat, rl, mu_rl, a


class SEPO(object):
    r"""
    A smoothed variant of EPO, with two adjustments for unrobust OOD objectives:
    a) normalization: unrobust OOD objective can yield large loss values that dominate the solutions of the LP,
                      hence we adopt the normalized OOD losses in the LP to resolve the issue
    b) regularization: solutions yielded by the LP can change sharply among steps, especially when switching descending phases 
                       hence we incorporate a L2 regularization in the LP to resolve the issue
    """
    def __init__(self, m, r, eps=1e-4, coe=0, verbose=False):
        # self.solver = cp.GLPK
        self.solver = cp.GUROBI
        # cvxopt.glpk.options["msg_lev"] = "GLP_MSG_OFF"
        self.m = m
        self.r = r/np.sum(r)
        self.eps = eps
        self.last_move = None
        self.a = cp.Parameter(m)        # Adjustments
        self.C = cp.Parameter((m, m))   # C: Gradient inner products, G^T G
        self.Ca = cp.Parameter(m)       # d_bal^TG
        self.rhs = cp.Parameter(m)      # RHS of constraints for balancing

        self.alpha = cp.Variable(m)     # Variable to optimize
        self.last_alpha = np.zeros_like(r)-1   
        self.coe = coe

        obj_bal = cp.Maximize(self.alpha @ self.Ca-self.coe*cp.sum_squares(self.alpha-self.last_alpha))   # objective for balance
        obj_bal_orig = cp.Maximize(self.alpha @ self.Ca)   # objective for balance
        constraints_bal = [self.alpha >= 0, cp.sum(self.alpha) == 1,  # Simplex
                           self.C @ self.alpha >= self.rhs]
        self.prob_bal = cp.Problem(obj_bal, constraints_bal)  # LP balance
        self.prob_bal_orig = cp.Problem(obj_bal_orig, constraints_bal)  # LP balance

        obj_dom = cp.Maximize(cp.sum(self.alpha @ self.C)-self.coe*cp.sum_squares(self.alpha-self.last_alpha))  # obj for descent
        constraints_res = [self.alpha >= 0, cp.sum(self.alpha) == 1,  # Restrict
                           self.alpha @ self.Ca >= -cp.neg(cp.max(self.Ca)),
                           self.C @ self.alpha >= 0]
        constraints_rel = [self.alpha >= 0, cp.sum(self.alpha) == 1,  # Relaxed
                           self.C @ self.alpha >= 0]
        self.prob_dom = cp.Problem(obj_dom, constraints_res)  # LP dominance
        self.prob_rel = cp.Problem(obj_dom, constraints_rel)  # LP dominance

        self.gamma = 0     # Stores the latest Optimum value of the LP problem
        self.mu_rl = 0     # Stores the latest non-uniformity

        self.verbose = verbose
        

    def get_alpha(self, l, G, r=None, C=False, get_mu=False):
        """calculate weights for all objectives given the gradient information

        Args:
            l (ndarray): the values of objective losses
            G (ndarray): inner products of the gradients of each objective loss w.r.t. params
            r (ndarray, optional): adopt this preference if specified
            C (bool, optional): True if the input gradients are inner products
            get_mu (bool, optional): return detailed information if True.

        Returns:
            alpha: the objective weights
            mu_rl (optional): the optimal value to the LP
            reset_optimizer (optional): whether to reset the inner optimizer
        """
        r = self.r if r is None else r
        assert len(l) == len(G) == len(r) == self.m, "length != m"
        l_hat, rl, self.mu_rl, self.a.value = self.adjustments(l, r)
        reset_optimizer = False
        if self.mu_rl <= 0.1:
            self.r[0]=max(1e-15,self.r[0]/10000)
            self.r = self.r/self.r.sum()
            print(f"pua preference {self.r}")
            l_hat, rl, self.mu_rl, self.a.value = self.adjustments(l, r)

        
        a_norm = np.linalg.norm(self.a.value)
        G_norm = np.linalg.norm(G,axis=1)
        Ga = G.T @ self.a.value
        self.C.value = G if C else G/np.expand_dims(G_norm,axis=1) @ G.T/a_norm
        self.Ca.value = G/np.expand_dims(G_norm,axis=1) @ Ga.T/a_norm

        if self.last_alpha.sum() is None:
            self.last_alpha = np.array(r)
        if self.mu_rl > self.eps:
            J = self.Ca.value > 0

            J_star_idx = np.where(rl == np.max(rl))[0]
            self.rhs.value = self.Ca.value.copy()
            # it's equivalent to setting no constraints to objectives in J
            # as maximize alpha^TCa would trivially satisfy the non-negativity
            self.rhs.value[J] = -np.inf     # Not efficient; but works.
            self.rhs.value[J_star_idx] =  max(0,self.Ca.value[J_star_idx]/2)
            
            if self.last_alpha.sum()<0:
                self.gamma = self.prob_bal_orig.solve(solver=self.solver, verbose=False)
            else:
                self.gamma = self.prob_bal.solve(solver=self.solver, verbose=False)

            self.last_move = "bal"
            
            if self.verbose:
                test_alpha = np.ones_like(self.a.value)/self.m
                print(self.last_alpha,self.C.value,self.Ca.value,self.rhs.value)
                print(self.gamma,test_alpha@self.Ca.value, self.alpha.value @ self.C.value)
                print(self.gamma,self.coe*np.linalg.norm(self.alpha.value-self.last_alpha)**2)
        else:
            self.gamma = self.prob_dom.solve(solver=self.solver, verbose=False)
            self.last_move = "dom"
        self.last_alpha = np.array(self.alpha.value)
        
        if get_mu:
            return self.alpha.value, self.mu_rl, reset_optimizer
        
        return self.alpha.value


    def mu(self, rl, normed=False):
        if len(np.where(rl < 0)[0]):
            raise ValueError(f"rl<0 \n rl={rl}")
            return None
        m = len(rl)
        l_hat = rl if normed else rl / rl.sum()
        eps = np.finfo(rl.dtype).eps
        l_hat = l_hat[l_hat > eps]
        return np.sum(l_hat * np.log(l_hat * m))


    def adjustments(self, l, r=1):
        m = len(l)
        rl = r * l
        
        l_hat = rl / rl.sum()
        mu_rl = self.mu(l_hat, normed=True)
        uniformity_div = np.log(l_hat * m) - mu_rl
        div_r = np.array(r)
        a = div_r * uniformity_div

        if self.verbose:
            print(a, rl, div_r, uniformity_div, l_hat, a.dot(l))
        return l_hat, rl, mu_rl, a


def getNumParams(params):
    numParams, numTrainable = 0, 0
    for param in params:
        npParamCount = np.prod(param.data.shape)
        numParams += npParamCount
        if param.requires_grad:
            numTrainable += npParamCount
    return numParams, numTrainable

def get_kl_div(losses, preference):
    pair_score = losses.dot(preference)
    return pair_score

def pair_selection(losses,val_accs,test_accs,anneal_iter=0,val_acc_bar=-1,pood=None):
    
    losses = losses[anneal_iter:]
    val_accs = val_accs[anneal_iter:]
    test_accs = test_accs[anneal_iter:]
    if val_acc_bar < 0:
        val_acc_bar = (np.max(val_accs)-np.min(val_accs))*0.05+np.min(val_accs)
    
    try:
        preference_base = 10**max(-12,int(np.log10(np.mean(losses[:,-1]))-2))
    except Exception as e:
        print(e)
        preference_base = 1e-12
    if len(losses[0])==2:
        preference = np.array([preference_base,1])
    elif len(losses[0])==4:
        preference = np.array([1e-12,1e-4,1e-2,1])
    elif len(losses[0])==5:
        preference = np.array([1e-12,1e-6,1e-4,1e-2,1])
    else:
        preference = np.array([1e-12,1e-2,1])
    
    if pood is not None:
        preference = pood
    print(f"Use preference: {preference}, validation acc bar: {val_acc_bar}")

    pair_score = np.array([get_kl_div(l,preference) if a>=val_acc_bar else 1e9 for (a,l) in zip(val_accs,losses)])
    sel_idx = np.argmin(pair_score)
    return sel_idx+anneal_iter, val_accs[sel_idx], test_accs[sel_idx]

def get_grad_sim(params,losses,preference=None,is_G=False,cosine=True):
    num_ood_losses = len(losses)-1
    if is_G:
        G = params
    else:
        pesudo_opt = SGD(params,lr=1e-6)
        grads = []
        for cur_loss in losses:
            pesudo_opt.zero_grad()
            cur_loss.backward(retain_graph=True)
            cur_grad = []
            for param in params:
                if param.grad is not None:
                    cur_grad.append(Variable(param.grad.data.clone().flatten(), requires_grad=False))
            # print(torch.cat(cur_grad).sum())
            grads.append(torch.cat(cur_grad))
        G = torch.stack(grads)
    if cosine:
        G = F.normalize(G,dim=1)
    GG = (G @ G.T).cpu()
    if preference is not None:
        G_weights = preference[1:]/np.sum(preference[1:])
    else:
        G_weights = np.ones(num_ood_losses)/num_ood_losses
    grad_sim =G_weights.dot(GG[0,1:]) 
    return grad_sim.item()
