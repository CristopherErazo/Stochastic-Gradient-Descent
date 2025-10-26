import numpy as np
from .utils import get_progress_bar
from .gradient import gradients
from .functions import activations
from .data import DataGenerator

class Trainer:
    def __init__(self, d, w_teacher, student, loss, lr, data_generator, N_walkers = None, normalize=True, spherical=True):
        """
        Initialize the Trainer with model parameters and training settings.
        Parameters:
        -----------
        d : int
            Dimensionality of the input data.
        w_teacher : (d,) array
            Weights of the teacher model.
        student : str
            Activation function for the student model.
        loss : str
            Loss function to use.
        lr : float
            Learning rate.
        data_generator : DataGenerator
            Instance of the DataGenerator class for sampling data.
        N_walkers : int or None
            If provided, number of parallel data streams (walkers) to train on. If None or 1, single chain training is used. 
            Must match data_generator.N_walkers if provided.
        normalize : bool
            Whether to normalize weights after each update.
        spherical : bool
            Whether to project gradients onto the sphere.
        """
        avail_losses = ["mse", "corr"]
        avail_activations = ["tanh", "relu", "He1", "He2", "He3", "He4", "He5","softmax", "He2+He3"] 
        if N_walkers is not None and N_walkers > 1:

            if data_generator.N_walkers is None or data_generator.N_walkers != N_walkers:
                raise ValueError("N_walkers must match data_generator.N_walkers when using multiple walkers.")
        else:

            N_walkers = None  # Ensure single chain mode
        if not isinstance(w_teacher, np.ndarray) or w_teacher.shape != (d,):
            raise ValueError(f"w_teacher must be a numpy array of shape ({d},)")
        if loss not in avail_losses:
            raise ValueError(f"Loss function '{loss}' not recognized. Available options: {avail_losses}")
        if student not in avail_activations:
            raise ValueError(f"Student activation '{student}' not recognized. Available options: {avail_activations}")
        if not  isinstance(data_generator, DataGenerator) :
            raise ValueError("data_generator must be an instance of the DataGenerator or SpikeGenerator class.")
        if data_generator.dim != d:
            raise ValueError("Data generator dimensionality does not match d.")
        if lr <= 0:
            raise ValueError("Learning rate must be positive.")
        
        self.d = d
        self.N_walkers = N_walkers
        self.student_fun, self.student_deriv = activations[student]
        self.lr = lr
        self.normalize = normalize
        self.spherical = spherical
        self.sample_data = data_generator.generate
        self.w_teacher = w_teacher
        if self.N_walkers is None or self.N_walkers == 1:
            self.gradient = gradients[loss]['single_chain']
        else:
            self.gradient = gradients[loss]['multi_chain']

    

    def evolution(self, w_initial, N_steps, progress=False , data_init = None):
        """
        Train the student model using gradient descent.
        """
        if self.N_walkers is None or self.N_walkers == 1:
            if not isinstance(w_initial, np.ndarray) or w_initial.shape != (self.d,):
                raise ValueError(f"For single chain training, w_initial must be a numpy array of shape ({self.d},)")
        else:
            if not isinstance(w_initial, np.ndarray) or w_initial.shape != (self.N_walkers, self.d):
                raise ValueError(f"For multi-chain training, w_initial must be a numpy array of shape ({self.N_walkers}, {self.d})")
            
        w_student = w_initial.copy()
        pbar = get_progress_bar(progress, N_steps)
        if data_init is not None:
            x , y = data_init
            flag = 1 if self.N_walkers is None or self.N_walkers == 1 else np.ones(self.N_walkers)
        else:
            x , y , flag = self.sample_data()

        for _ in range(N_steps):
            grad = self.gradient(w_student, x, y, self.student_fun, self.student_deriv, self.spherical)
            w_student -= self.lr * grad / self.d

            if self.normalize: 
                w_student /= np.linalg.norm(w_student, axis=-1, keepdims=True)
            
            yield w_student , flag , grad
            pbar.update(1)  
            x , y , flag = self.sample_data()    
        pbar.close()
