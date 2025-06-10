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

# chosen
inner_itr = 200
c_prime = 1
ro = 100
l = 200
S = 10
S_k = 1


def run_task(seed):
    @wrap_experiment(archive_launch_repo=False,
                     log_dir="/root/Data/icml/final/reacher_vr_klscrn_itr={}c-prime={}ro={}l={}S={}seed={}S_k={}".format(
                         inner_itr,
                         c_prime,
                         ro,
                         l,
                         S, seed,
                         S_k
                         ))
    def srcn_reacher(ctxt=None, seed=None):
        """
        Args:
            ctxt (garage.experiment.ExperimentContext): The experiment
                configuration used by Trainer to create the snapshotter.
            seed (int): Used to seed the random number generator to produce
                determinism.

        """
        n_epochs = 4000
        sampler_batch_size = 10000

        set_seed(seed)
        env = GymEnv('Reacher-v2')
        env._env.seed(seed)
        env.action_space.seed(seed)
        trainer = Trainer(ctxt)

        policy = GaussianMLPPolicy(env.spec,
                                   hidden_sizes=[64, 64],
                                   hidden_nonlinearity=torch.tanh,
                                   output_nonlinearity=None)

        # value_function = GaussianMLPValueFunction(env_spec=env.spec,
        #                                           hidden_sizes=(32, 32),
        #                                           hidden_nonlinearity=torch.tanh,
        #                                           output_nonlinearity=None)
        value_function = LinearFeatureBaseline(env_spec=env.spec)

        sampler = RaySampler(agents=policy,
                             envs=env,
                             max_episode_length=env.spec.max_episode_length,
                             n_workers=40)

        policy_optimizer = OptimizerWrapper((VRSCRNOptimizer, {
            "inner_itr": inner_itr, "c_prime": c_prime, "ro": ro, "l": l, 'S': S, 'S_k': S_k,

        }), policy)

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

    srcn_reacher(seed=seed)


seeds = [4, 49, ]
# seeds = [22, 46, 1, 7, 31,]
for seed in seeds:
    run_task(seed)
