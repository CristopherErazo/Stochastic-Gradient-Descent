"""
Module for SGD optimizers.
Contains implementations of various Stochastic Gradient Descent (SGD) optimizers
that return updated weights based on computed gradients, data samples, and learning rates.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple
from .data import DataGenerator
from .gradient import gradients
from .functions import get_activation
import numpy as np


class BaseOptimizer(ABC):
    """
    Simple abstract optimizer that caches:
      - gradient callable
      - data_generator handle
      - student activation pair (cached)
      - fixed lr and optional rho (set at construction)
    Per-step API: step(w) -> (w_new, flag)
    """
    def __init__(
                self,
                data_generator: DataGenerator,
                loss: str,
                student_spec: Any,
                lr: float,
                rho: Optional[float] = None,
                spherical: bool = True):

        avail_losses = ["mse", "corr"]
        if loss not in avail_losses:
            raise ValueError(f"Loss function '{loss}' not recognized. Available options: {avail_losses}")
        if lr is None:
            raise ValueError("lr must be provided to BaseOptimizer")
        if not isinstance(data_generator, DataGenerator):
            raise TypeError("data_generator must be a DataGenerator")
        self.sample_data = data_generator.generator
        self.student_fun, self.student_deriv = get_activation(student_spec)
        self.dim = data_generator.dim
        self.lr = lr / self.dim
        self.rho = rho / self.dim if rho is not None else None
        self.spherical = spherical
        self.N_walkers = data_generator.N_walkers
        self.loss = loss
        if self.N_walkers is None or self.N_walkers == 1:
            self.gradient = gradients[self.loss]['single_chain']
        else:
            self.gradient = gradients[self.loss]['multi_chain']

  
    @abstractmethod
    def step(self, w: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Any]:
        """
        Perform a single optimization step.

        Parameters
        ----------
        w : array_like
            Current weights of the model. Shape (d,) or (N_walkers, d).

        Returns
        -------
        w_new : ndarray
            Updated weights after the optimization step. Shape (d,) or (N_walkers, d).
        grad : ndarray
            Computed gradient used for the update. Shape (d,) or (N_walkers, d).
        flag : ndarray
            Flags indicating sampling source. Shape (N_walkers,); 1 if sampled from dataset, 0 if fresh.
        """
        raise NotImplementedError
        

class SingleStep(BaseOptimizer):
    """Standard SGD single-step using fixed lr."""
    def step(self, w: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Any]:
        x, y, flag = self.sample_data()
        grad = self.gradient(w, x, y, self.student_fun, self.student_deriv, self.spherical)
        return w - self.lr * grad, grad, flag


class ExtraGradient(BaseOptimizer):
    """Extra-gradient (two-stage) optimizer. Requires rho set at construction."""
    def step(self, w: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Any]:
        if self.rho is None:
            raise ValueError("ExtraGradient requires rho to be set at construction")
        x, y, flag = self.sample_data()
        grad = self.gradient(w, x, y, self.student_fun, self.student_deriv, self.spherical)
        w_aux = w - self.rho * grad
        grad_aux = self.gradient(w_aux, x, y, self.student_fun, self.student_deriv, self.spherical)
        return w - self.lr * grad_aux, grad , flag


class TwiceStep(BaseOptimizer):
    """Two sequential SGD evaluations using the same sample (fixed lr)."""
    def step(self, w: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Any]:
        x, y, flag = self.sample_data()
        grad = self.gradient(w, x, y, self.student_fun, self.student_deriv, self.spherical)
        w_aux = w - self.lr * grad
        grad_aux = self.gradient(w_aux, x, y, self.student_fun, self.student_deriv, self.spherical)
        return w_aux - self.lr * grad_aux, grad, flag
