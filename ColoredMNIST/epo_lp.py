import numpy as np
import cvxpy as cp
import cvxopt

from scipy.special import softmax
class EPO_LP(object):

    def __init__(self, m, n, r, eps=1e-4, softmax_norm=False):
        # self.solver = cp.GLPK
        self.solver = cp.GUROBI
        # cvxopt.glpk.options["msg_lev"] = "GLP_MSG_OFF"
        self.m = m
        self.n = n
        self.r = r
        self.eps = eps
        self.last_move = None
        self.a = cp.Parameter(m)        # Adjustments
        self.C = cp.Parameter((m, m))   # C: Gradient inner products, G^T G
        self.Ca = cp.Parameter(m)       # d_bal^TG
        self.rhs = cp.Parameter(m)      # RHS of constraints for balancing

        self.alpha = cp.Variable(m)     # Variable to optimize

        obj_bal = cp.Maximize(self.alpha @ self.Ca)   # objective for balance
        constraints_bal = [self.alpha >= 0, cp.sum(self.alpha) == 1,  # Simplex
                           self.C @ self.alpha >= self.rhs]
        self.prob_bal = cp.Problem(obj_bal, constraints_bal)  # LP balance

        obj_dom = cp.Maximize(cp.sum(self.alpha @ self.C))  # obj for descent
        constraints_res = [self.alpha >= 0, cp.sum(self.alpha) == 1,  # Restrict
                           self.alpha @ self.Ca >= -cp.neg(cp.max(self.Ca)),
                           self.C @ self.alpha >= 0]
        constraints_rel = [self.alpha >= 0, cp.sum(self.alpha) == 1,  # Relaxed
                           self.C @ self.alpha >= 0]
        self.prob_dom = cp.Problem(obj_dom, constraints_res)  # LP dominance
        self.prob_rel = cp.Problem(obj_dom, constraints_rel)  # LP dominance

        self.gamma = 0     # Stores the latest Optimum value of the LP problem
        self.mu_rl = 0     # Stores the latest non-uniformity

        self.softmax_norm = softmax_norm # use which normalization to calc. non-uniformity

    def get_alpha(self, l, G, r=None, C=False, relax=False):
        r = self.r if r is None else r
        assert len(l) == len(G) == len(r) == self.m, "length != m"

        if self.softmax_norm:
            r = np.exp(r)
            l = np.exp(l)
        rl, self.mu_rl, self.a.value = self.adjustments(l, r)

        self.C.value = G if C else G @ G.T
        self.Ca.value = self.C.value @ self.a.value

        if self.mu_rl > self.eps:
            J = self.Ca.value > 0
            # if len(np.where(J)[0]) > 0:
            if True:
                J_star_idx = np.where(rl == np.max(rl))[0]
                self.rhs.value = self.Ca.value.copy()
                self.rhs.value[J] = -np.inf     # Not efficient; but works.
                self.rhs.value[J_star_idx] = 0
            else:
                self.rhs.value = np.zeros_like(self.Ca.value)
            self.gamma = self.prob_bal.solve(solver=self.solver, verbose=False,reoptimize=True)
            self.last_move = "bal"
        else:
            if relax:
                self.gamma = self.prob_rel.solve(solver=self.solver, verbose=False,reoptimize=True)
            else:
                self.gamma = self.prob_dom.solve(solver=self.solver, verbose=False,reoptimize=True)
            self.last_move = "dom"
        return self.alpha.value


    def mu(self, rl, normed=False):
        if len(np.where(rl < 0)[0]):
            raise ValueError(f"rl<0 \n rl={rl}")
            return None
        m = len(rl)
        if normed:
            # if self.softmax_norm:
            #     l_hat = softmax(rl)
            # else:
            l_hat = rl/rl.sum()
        # l_hat = rl if normed else rl / rl.sum()
        eps = np.finfo(rl.dtype).eps
        l_hat = l_hat[l_hat > eps]
        return np.sum(l_hat * np.log(l_hat * m))


    def adjustments(self, l, r=1):
        m = len(l)
        rl = r * l
        # if self.softmax_norm:
        #     l_hat = softmax(rl)
        # else:
        #     l_hat = rl / rl.sum()
        # rl = np.exp(rl) if self.softmax_norm else rl
        l_hat = rl/rl.sum()
        # print(l_hat[0]/l_hat[2])
        mu_rl = self.mu(l_hat, normed=True)
        a = r * (np.log(l_hat * m) - mu_rl)
        return rl, mu_rl, a

# def get_param_dim(model):
#     for param in model.parameters():
#         if param.grad is not None:
#             cur_grad.append(Variable(param.data.clone().flatten(), requires_grad=False))
#     grads.append(torch.cat(cur_grad))


def getNumParams(params):
    numParams, numTrainable = 0, 0
    for param in params:
        npParamCount = np.prod(param.data.shape)
        numParams += npParamCount
        if param.requires_grad:
            numTrainable += npParamCount
    return numParams, numTrainable
