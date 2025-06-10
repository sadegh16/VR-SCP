#!/usr/bin/env python3

from garage import wrap_experiment
from garage.envs import GymEnv
from garage.experiment.deterministic import set_seed
from garage.np.baselines import LinearFeatureBaseline
from garage.sampler import RaySampler
from garage.torch.algos import VRSCRN
from garage.torch.optimizers import OptimizerWrapper
from garage.torch.optimizers.VR_SCRN_optimizer import VRSCRNOptimizer
from garage.torch.policies import GaussianMLPPolicy
from garage.trainer import Trainer

import torch

inner_itr = 1000
c_prime = 1
ro = 1
l = 400
S = 5
S_k = 1


def run_task(seed):
    @wrap_experiment(archive_launch_repo=False,
                     log_dir="/root/Data/icml/final/humanoid_vr_klscrn_itr={}c-prime={}ro={}l={}S={}seed={}S_k={}".format(
                         inner_itr,
                         c_prime, ro, l,
                         S, seed, S_k

                     ))
    def srcn_humanoid(ctxt=None, seed=1):
        n_epochs = 6200
        sampler_batch_size = 10000

        set_seed(seed)
        env = GymEnv('Humanoid-v2')
        env._env.seed(seed)
        env.action_space.seed(seed)
        trainer = Trainer(ctxt)

        policy = GaussianMLPPolicy(env.spec,
                                   hidden_sizes=[64, 64],
                                   hidden_nonlinearity=torch.tanh,
                                   output_nonlinearity=None)

        value_function = LinearFeatureBaseline(env_spec=env.spec)

        sampler = RaySampler(agents=policy,
                             envs=env,
                             max_episode_length=500)

        policy_optimizer = OptimizerWrapper((VRSCRNOptimizer, {
            "inner_itr": inner_itr, "c_prime": c_prime, "ro": ro, "l": l, 'S': S, 'S_k': S_k, 'kl_limit': 1e-1,

        }), policy)

        # policy_optimizer = OptimizerWrapper((SGD, {
        #     "lr": 0.01,
        # }), policy)

        algo = VRSCRN(env_spec=env.spec,
                      policy=policy,
                      value_function=value_function,
                      sampler=sampler,
                      discount=0.99,
                      center_adv=False,
                      policy_optimizer=policy_optimizer,
                      neural_baseline=False,
                      )

        trainer.setup(algo, env)
        trainer.train(n_epochs=n_epochs, batch_size=sampler_batch_size)

    srcn_humanoid(seed=seed)


seeds = [14, 33, 3, 4, 49, ]
# seeds = [22, 46, 1, 7, 31, ]

for seed in seeds:
    run_task(seed=seed)
