import numpy as np
from dowel import logger, tabular
from garage.torch import global_device
from torch.optim import Optimizer

import torch


class VRSCRNOptimizer(Optimizer):
    def __init__(self, params, inner_itr=10, ro=0.1, l=0.5, epsilon=1e-3, c_prime=0.1, K=10,
                 C=10, S=5, S_k=1, kl_limit=1e-2):

        self.ro = ro
        self.l = l
        self.S = S
        self.epsilon = epsilon
        self.c_prime = c_prime
        self.inner_itr = inner_itr
        self.step_size = 1 / (20 * l)
        self.iteration = -1
        self.sqr_grads_norms = 0
        self.last_grad_norm = 0
        self.C = C
        self.power = 1.0 / 3.0
        self.K = 1.0 * K
        self.S_k = S_k
        self.soft_checkpoint = 0
        self._max_backtracks = 30
        self._backtrack_ratio = 0.8
        self._max_constraint_value = 0.01
        self.last_skipped = False
        self.kl_limit = kl_limit
        self.line_search_ok=0

        defaults = dict()
        super(VRSCRNOptimizer, self).__init__(params, defaults)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['last_point'] = torch.zeros_like(p)
                state['current_point'] = torch.zeros_like(p)

    def compute_norm_of_list_var(self, array_):
        """
        Args:
        param array_: list of tensors
        return:
        norm of the flattened list
        """
        norm_square = 0
        for i in range(len(array_)):
            norm_square += array_[i].norm(2).item() ** 2
        return norm_square ** 0.5

    def inner_product_of_list_var(self, array1_, array2_):

        """
        Args:
        param array1_: list of tensors
        param array2_: list of tensors
        return:
        The inner product of the flattened list
        """

        sum_list = 0
        for i in range(len(array1_)):
            sum_list += torch.sum(array1_[i] * array2_[i])
        return sum_list

    def save_current_point(self, group):

        with torch.no_grad():
            for p in group['params']:
                state = self.state[p]
                current_point = state['current_point']
                current_point.copy_(p)

    def update_model_to_random_line_point(self, b, group):
        """
        update the parameter based on the displacement
        """

        with torch.no_grad():
            for p in group['params']:
                state = self.state[p]
                last_point, current_point = state['last_point'], state['current_point']
                p.copy_((1 - b) * current_point + b * last_point)

    def update_model_to_current_point(self, group):
        """
        update the parameter based on the displacement
        """

        with torch.no_grad():
            for p in group['params']:
                state = self.state[p]
                p.copy_(state['current_point'])

    def cubic_subsolver(self, g, grads, g_ll, param, g_norm: float, epsilon: float,
                        ro: float, l: float, group, compute_kl, compute_loss):
        """
        solve the sub problem with gradient decent
        """
        deltas = [0] * len(grads)
        last_deltas = [None] * len(grads)
        g_tildas = [0] * len(grads)

        sigma = self.c_prime * (epsilon * ro) ** 0.5 / l
        print("**************", sigma)
        for i in range(len(g)):
            deltas[i] = torch.zeros(g[i].shape).to(global_device())
            last_deltas[i] = torch.zeros(g[i].shape).to(global_device())
            khi = torch.rand(g[i].shape).to(global_device())
            g_tildas[i] = g[i].clone() + sigma * khi
        for t in range(self.inner_itr):
            # compute hessian vector with respect to delta
            hvp = self.compute_hvp(grads, g_ll, param, deltas)
            deltas_norm = self.compute_norm_of_list_var(deltas)
            for i in range(len(g)):
                last_deltas[i]= deltas[i].clone()
                deltas[i] = deltas[i] - self.step_size * (
                        g_tildas[i] + hvp[i].clone() + ro / 2 * deltas_norm * deltas[i])

                deltas[i].detach()
            kl_value=compute_kl(deltas)
            if kl_value > self.kl_limit:
                break
            ######

        # compute hessian vector with respect to delta
        hvp = self.compute_hvp(grads, g_ll, param, last_deltas)
        deltas_norm = self.compute_norm_of_list_var(last_deltas)
        delta_m = 0
        for i in range(len(grads)):
            delta_m += torch.sum(grads[i] * last_deltas[i]) + 0.5 * torch.sum(
                last_deltas[i] * hvp[i].clone()) + ro / 6 * deltas_norm ** 3

        return delta_m.item(), deltas_norm ** 0.5, last_deltas

    def update_parameters(self, group, deltas, ratio):
        i = 0
        with torch.no_grad():
            for p in group['params']:
                state = self.state[p]
                state['last_point'].copy_(p)
                p.add_(ratio * deltas[i].clone())
                i += 1

    def compute_multi_points_hvp(self, closure, vector, group):
        hvps = None
        for i in range(self.S_k - 1, -1, -1):
            # compute hessian vector
            self.update_model_to_random_line_point(i / self.S_k, group)
            with torch.enable_grad():
                g_ll = closure(batch_rate=int((self.iteration) ** (2. / 3)))
            modified_grads = []
            modified_param = []
            for p in group['params']:
                modified_grads.append(p.grad)
                modified_param.append(p)
            # compute hessian vector product
            hvp = self.compute_hvp(modified_grads, g_ll, modified_param, vector)

            if hvps is None:
                hvps = list(hvp)
            else:
                for j in range(len(hvps)):
                    hvps[j] += hvp[j].clone() / self.S_k

        return hvps, g_ll

    def compute_hvp(self, grads, g_ll, params, vector):
        # compute first term
        inner_product = self.inner_product_of_list_var(g_ll, vector)
        # compute second term
        second_term = torch.autograd.grad(outputs=grads, inputs=params, grad_outputs=vector,
                                          retain_graph=True)
        hessian_vector_product = []
        for i in range(len(grads)):
            hessian_vector_product.append(inner_product * grads[i] + second_term[i])
        return hessian_vector_product

    def step(self, closure=None, compute_kl=None, compute_loss=None):
        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        g_ll = None
        self.iteration += 1
        g_square_norm = 0
        vector = []
        grads = []
        g = []
        param = []
        for group in self.param_groups:
            self.save_current_point(group)
            for p in group['params']:
                vector.append((self.state[p]['current_point'].detach() - self.state[p]['last_point'].detach()).to(
                    global_device()))

            if self.iteration % self.S != 0:
                hvp, g_ll = self.compute_multi_points_hvp(closure, vector, group)
            else:
                with torch.enable_grad():
                    g_ll = closure(count_samples=True)

            for p in group['params']:
                grads.append(p.grad)
                param.append(p)
            # compute gradiant vector
            i = 0
            for p in group['params']:
                state = self.state[p]

                if 'momentum_buffer' not in state or self.iteration % self.S == 0:
                    buf = state['momentum_buffer'] = grads[i]
                else:
                    buf = state['momentum_buffer'].detach()
                    if self.last_skipped:
                        buf.add_(hvp[i].detach() + grads[i])
                    else:
                        buf.add_(hvp[i].detach())

                g.append(buf)
                g_square_norm += buf.norm(2).item() ** 2
                i += 1

            if self.iteration % self.S != 0 and self.last_skipped:
                self.soft_checkpoint += 1
                self.last_skipped = False
            # store square of grad norm
            self.sqr_grads_norms += self.last_grad_norm
            delta_m, deltas_norm, deltas = self.cubic_subsolver(g, grads, g_ll, param, g_square_norm ** 0.5,
                                                                self.epsilon,
                                                                self.ro,
                                                                self.l, group, compute_kl, compute_loss)
            if delta_m is not None:
                self.update_parameters(group, deltas, 1)
                with tabular.prefix("VR-SCRN" + '/'):
                    tabular.record('delta of m', delta_m)
                    tabular.record('norm of deltas', deltas_norm)
                    tabular.record('soft_checkpoint', self.soft_checkpoint)
                    tabular.record('line_search_ok', self.line_search_ok)
                    logger.log(tabular)
            return None

