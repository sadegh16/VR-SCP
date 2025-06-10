import torch
from torch.optim import Optimizer
from dowel import logger, tabular


class VRSCRNOptimizer(Optimizer):
    def __init__(self, params, inner_itr=10, ro=0.1, l=0.5, epsilon=1e-3, c_prime=0.1, K=10,
                 C=10, S=5, S_k=1):

        self.ro = ro
        self.l = l
        self.S = S
        self.epsilon = epsilon
        self.c_prime = c_prime
        self.inner_itr = inner_itr
        self.step_size = 1 / (20 * l)
        self.iteration = -1
        defaults = dict()
        self.sqr_grads_norms = 0
        self.last_grad_norm = 0
        self.C = C
        self.power = 1.0 / 3.0
        self.K = 1.0 * K
        self.eta = 1.0 / self.C
        self.alpha = 2 * self.K * self.eta ** 2
        self.S_k = S_k
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
                current_point.copy_(p.clone())

    def update_model_to_random_line_point(self, b, group):
        """
        update the parameter based on the displacement
        """
        print("**********************", b)

        with torch.no_grad():
            for p in group['params']:
                state = self.state[p]
                last_point, current_point = state['last_point'], state['current_point']
                p.copy_(b * current_point + (1 - b) * last_point)



    def update_model_to_current_point(self, group):
        """
        update the parameter based on the displacement
        """

        with torch.no_grad():
            for p in group['params']:
                state = self.state[p]
                p.copy_(state['current_point'].clone())

    def cubic_subsolver(self, g, grads, param, g_norm: float, epsilon: float, ro: float, l: float):
        """
        solve the sub problem with gradient decent
        """
        deltas = [0] * len(grads)
        g_tildas = [0] * len(grads)

        with torch.no_grad():
            if g_norm >= l ** 2 / self.ro:
                # compute hessian vector with respect to grads
                hvp = torch.autograd.grad(outputs=grads, inputs=param,
                                          grad_outputs=g, retain_graph=True)
                g_t_dot_bg_t = self.inner_product_of_list_var(g, hvp) / (ro * (g_norm ** 2))
                R_c = -g_t_dot_bg_t + (g_t_dot_bg_t ** 2 + 2 * g_norm / ro) ** 0.5
                for i in range(len(g)):
                    deltas[i] = -R_c * g[i].clone() / g_norm

            else:
                sigma = self.c_prime * (epsilon * ro) ** 0.5 / l
                for i in range(len(g)):
                    deltas[i] = torch.zeros(g[i].shape)
                    khi = torch.rand(g[i].shape)
                    g_tildas[i] = g[i].clone() + sigma * khi
                for t in range(self.inner_itr):
                    # compute hessian vector with respect to delta
                    hvp = torch.autograd.grad(outputs=grads, inputs=param,
                                              grad_outputs=deltas, retain_graph=True)
                    deltas_norm = self.compute_norm_of_list_var(deltas)
                    if self.compute_norm_of_list_var(hvp) > 200:
                        break

                    for i in range(len(g)):
                        deltas[i] = deltas[i] - self.step_size * (
                                g_tildas[i] + hvp[i].clone() + ro / 2 * deltas_norm * deltas[i])

        # compute hessian vector with respect to delta
        hvp = torch.autograd.grad(outputs=grads, inputs=param,
                                  grad_outputs=deltas, retain_graph=True)
        deltas_norm = self.compute_norm_of_list_var(deltas)
        delta_m = 0
        for i in range(len(grads)):
            delta_m += torch.sum(grads[i] * deltas[i]) + 0.5 * torch.sum(deltas[i] * hvp[i].clone()) + ro / 6 * deltas_norm ** 3

        deltas_norm = 0
        # update the displacement
        for group in self.param_groups:
            i = 0
            for p in group['params']:
                state = self.state[p]
                deltas_norm += deltas[i].norm(2).item() ** 2
                state['displacement'] = deltas[i]
                i += 1

        return delta_m.item(), deltas_norm ** 0.5

    def cubic_finalsolver(self, grads, param, epsilon: float, ro: float, l: float):
        """
        solve the sub problem with gradient decent
        """
        grads_m = [0] * len(grads)
        with torch.no_grad():
            deltas = [0] * len(grads)
            for i in range(len(grads)):
                deltas[i] = torch.zeros_like(grads[i])
                grads_m[i] = grads[i].clone()
            while self.compute_norm_of_list_var(grads_m, ) > epsilon / 2:
                hvp = torch.autograd.grad(outputs=grads, inputs=param, grad_outputs=deltas, retain_graph=True)
                for i in range(len(grads)):
                    deltas[i] = deltas[i] - self.step_size * grads_m[i]
                deltas_norm = self.compute_norm_of_list_var(deltas)
                for i in range(len(grads)):
                    grads_m[i] = grads[i] + hvp[i] + ro / 2 * deltas_norm * deltas[i]

            # update the displacement
            for group in self.param_groups:
                with torch.no_grad():
                    i = 0
                    for p in group['params']:
                        state = self.state[p]
                        state['displacement'] = deltas[i]
                        i += 1

    def update_parameters(self, ):

        for group in self.param_groups:
            with torch.no_grad():
                for p in group['params']:
                    state = self.state[p]
                    displacement = state['displacement']
                    state['last_point'].copy_(p)
                    p.add_(displacement.clone())

    def compute_multi_points_hvp(self, closure, vector, group):
        hvps = None
        for i in range(self.S_k):
            shrink = 1
            # compute hessian vector
            self.update_model_to_random_line_point(i / self.S_k, group)
            with torch.enable_grad():
                closure()
            modified_grads = []
            modified_param = []
            for p in group['params']:
                if p.grad is None:
                    continue
                modified_grads.append(p.grad)
                modified_param.append(p)
            hvp = torch.autograd.grad(outputs=modified_grads, inputs=modified_param, grad_outputs=vector,)
            norm_of_hvp=self.compute_norm_of_list_var(hvp)
            print(norm_of_hvp)
            # if norm_of_hvp > 10:
            #     shrink=50/norm_of_hvp

            if hvps is None:
                hvps = list(hvp)
            else:
                for j in range(len(hvps)):
                    hvps[j] += hvp[j].clone()
        for i in range(len(hvps)):
            hvps[i] /= (self.S_k)
        return hvps

    def step(self, closure=None):
        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        self.iteration += 1

        g_square_norm = 0
        grad_square_norm = 0

        # update eta, alpha
        if self.iteration > 1:
            new_eta = 1. / (self.C * self.iteration ** self.power)
            self.alpha = 2 * self.K * self.eta * new_eta
            self.eta = new_eta

        vector = []
        grads = []
        g = []
        param = []
        for group in self.param_groups:
            self.save_current_point(group)
            for p in group['params']:
                vector.append(self.state[p]['current_point'].detach() - self.state[p]['last_point'].detach())
            if self.iteration >= 1:
                hvps = self.compute_multi_points_hvp(closure, vector, group)
                self.update_model_to_current_point(group)

            with torch.enable_grad():
                closure()
            for p in group['params']:
                if p.grad is None:
                    continue
                grads.append(p.grad)
                param.append(p)
            # compute gradiant vector
            i = 0
            for p in group['params']:
                state = self.state[p]
                d_p = grads[i]
                grad_square_norm += d_p.norm(2).item() ** 2

                if 'momentum_buffer' not in state or self.iteration % self.S == 0:
                    buf = state['momentum_buffer'] = d_p
                else:
                    buf = state['momentum_buffer'].detach()
                    buf.add_(hvps[i])

                d_p = buf
                g.append(d_p)
                g_square_norm += d_p.norm(2).item() ** 2
                i += 1

        # store square of grad norm
        self.sqr_grads_norms += self.last_grad_norm
        self.last_grad_norm = grad_square_norm
        delta_m, deltas_norm = self.cubic_subsolver(g, grads, param, g_square_norm ** 0.5, self.epsilon, self.ro,
                                                    self.l)
        self.update_parameters()

        with tabular.prefix("VR-SCRN" + '/'):
            tabular.record('delta of m', delta_m)
            tabular.record('norm of gradient', grad_square_norm ** (1. / 2))
            tabular.record('norm of deltas', deltas_norm)
            # tabular.record('landa min', lambda_min)
            logger.log(tabular)
        return None
