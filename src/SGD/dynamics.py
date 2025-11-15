import numpy as np
from .utils import get_progress_bar
from .optimizers import BaseOptimizer

class Trainer:
    def __init__(self, 
                d: int, 
                w_teacher: np.ndarray,
                optimizer: BaseOptimizer,
                N_walkers : int = None,
                normalize : bool =True):
        """
        Initialize the Trainer with model parameters and training settings.

        Parameters
        ----------
        d : int
            Dimensionality of the input data.
        w_teacher : ndarray, shape (d,)
            Weights of the teacher model.
        optimizer : BaseOptimizer
            Optimizer instance to use for training.
        N_walkers : int or None, optional
            Number of parallel data streams (walkers). If None or 1, single-chain mode is used.
        normalize : bool, optional
            Whether to normalize weights after each update (default True).
        """
        
        if not isinstance(optimizer, BaseOptimizer):
            raise TypeError("optimizer must be an instance of BaseOptimizer.")
        if N_walkers is None or N_walkers == 1:
            N_walkers = None  # Ensure single chain mode

        if not isinstance(w_teacher, np.ndarray) or w_teacher.shape != (d,):
            raise ValueError(f"w_teacher must be a numpy array of shape ({d},)")
        

        self.d = d
        self.N_walkers = N_walkers
        self.optimizer = optimizer
        self.normalize = normalize
        self.w_teacher = w_teacher


    

    def evolution(self, w_initial, N_steps, progress=False):
        """
        Train the student model using gradient descent.

        Parameters
        -----------
        w_initial : ndarray
            Initial weights of the student model. Shape (d,) or (N_walkers, d).
        N_steps : int
            Number of training steps to perform.
        progress : bool
            Whether to display a progress bar.

    
        Yields 
        -------
        w_student : ndarray
            Updated weights of the student model after each step. Shape (d,) or (N_walkers, d).
        flag : ndarray
            Additional information from the optimizer step.
        grad : ndarray
            Computed gradient used for the update. Shape (d,) or (N_walkers, d).
        """
        if self.N_walkers is None or self.N_walkers == 1:
            if not isinstance(w_initial, np.ndarray) or w_initial.shape != (self.d,):
                raise ValueError(f"For single chain training, w_initial must be a numpy array of shape ({self.d},)")
        else:
            if not isinstance(w_initial, np.ndarray) or w_initial.shape != (self.N_walkers, self.d):
                raise ValueError(f"For multi-chain training, w_initial must be a numpy array of shape ({self.N_walkers}, {self.d})")
            
        w_student = w_initial.copy()
        pbar = get_progress_bar(progress, N_steps)


        for _ in range(N_steps):
            w_student, grad, flag = self.optimizer.step(w_student)

            if self.normalize: 
                w_student /= np.linalg.norm(w_student, axis=-1, keepdims=True)
            
            yield w_student , flag , grad
            pbar.update(1)    
        pbar.close()
