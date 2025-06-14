"""PyTorch optimizers."""
from garage.torch.optimizers.conjugate_gradient_optimizer import (
    ConjugateGradientOptimizer)
from garage.torch.optimizers.differentiable_sgd import DifferentiableSGD
from garage.torch.optimizers.differentiable_sgd import DifferentiableSGD
from garage.torch.optimizers.optimizer_wrapper import OptimizerWrapper
from garage.torch.optimizers.VR_SCRN_optimizer import VRSCRNOptimizer
from garage.torch.optimizers.SGD_optimizer import SGD
__all__ = [
    'OptimizerWrapper', 'ConjugateGradientOptimizer', 'DifferentiableSGD',
    'VRSCRNOptimizer', 'SGD',
]
