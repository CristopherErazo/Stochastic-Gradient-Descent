"""
This module contains a class for data generation .
"""

# python
from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, Tuple
# from .functions import activations
from .functions import get_activation


# ##########################################################
# Abstract base class for input samplers
##########################################################
class DataSampler(ABC):
    """Model-specific sampler: returns X (n,d) and optional y (n,)."""
    def __init__(self, dim:int, rng:Optional[np.random.Generator]=None):
        self.dim = dim
        self._rng = np.random.default_rng() if rng is None else rng

    @abstractmethod
    def sample(self, n:int) -> Tuple[np.ndarray, np.ndarray]:
        """Return (X, y) . X shape (n, dim)."""


class Perceptron(DataSampler):
    """
    Single perceptron model y = g(w_teacher @ x) + eps with random Gaussian inputs and noise.
    """ 
    def __init__(self, dim, w_teacher:Optional[np.ndarray]=None, teacher:str='He3', noise:Optional[float]=0.0, rng:Optional[np.random.Generator]=None):
        """
        Parameters:
        -----------
        dim : int
            input dimension
        w_teacher : array (dim,), optional
            teacher weight vector. If None, sample randomly from S(dim-1)
        teacher : str or list of names or list of (name,weight) pairs or dict
            activation function name for the teacher
        noise : float, optional
            standard deviation of additive Gaussian noise
        rng : numpy generator, optional
            random number generator. If None, instantiate a new one
        """
        super().__init__(dim, rng)
        # make sure given dimension is consistent with input dimension
        if w_teacher is not None and len(w_teacher) != dim:
            raise ValueError(f"Teacher vector needs to have dimension {dim}, has dimension {len(w_teacher)}.")
        
        if w_teacher is None:
            self.w_teacher = self._rng.normal(size=(dim,))
            self.w_teacher /= np.linalg.norm(self.w_teacher)
        else:
            self.w_teacher = w_teacher
        self.noise = noise
        self.teacher_fun = get_activation(teacher)[0]

    
    def sample(self, n:int=1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample n data points from the perceptron model.
        Parameters:
        -----------
        n : int
            number of samples
        Returns:
        --------
        x : array (n, dim)
            input data
        y : array (n,)
            output data
        """
        x = self._rng.normal(size=(n, self.dim))
        y = self.teacher_fun(x @ self.w_teacher) + self.noise * self._rng.normal(size=(n,))
        return x, y
        


class SpikedCumulant(DataSampler):
    """
    Spiked cumulant model with arbitrary latent variable.
    """

    def __init__(self, dim , spike:Optional[np.ndarray]=None, snr:float=5.0, rng:Optional[np.random.Generator]=None, whiten:bool=True, p_spike:float=0.5):
        """
        Parameters:
        -----------
        dim : int
            input dimension
        spike : array (dim,)
            contains the spike. If None, will sample one uniformly from S(dim-1)
        snr : float
            signal-to-noise ratio, prefactor of the spike term
        rng : numpy generator
            random number generator. If None, instantiate a new one
        whiten : bool
            if True, will whiten the inputs with average covariance matrix. Default: True.
        p_spike : float
            probability of including the spike in each sample
        """
        super().__init__(dim, rng)

        # make sure given dimension is consistent with input dimension
        if spike is not None and len(spike) != dim:
            raise ValueError(f"Spike needs to have dimension {dim}, has dimension {len(spike)}.")

        if spike is None:
            self.spike = self._rng.normal(size=(dim,))
            self.spike /= np.linalg.norm(self.spike)
        else:
            self.spike = spike
    
        self.dim = dim
        self.snr = snr
        self.whiten = whiten
        self.p_spike = p_spike

        # Compute the whitening matrix (square root of inv of cov)
        uuT = np.outer(self.spike, self.spike)
        self._S = np.eye(dim) - snr / (1 + snr + np.sqrt(1 + snr)) * uuT

    @abstractmethod
    def _sample_latents(self, n:int) -> np.ndarray:
        """Samples the distribution over the latent variable for this spiked cumulant model.

        Parameters
        ----------
        n : int
            number of samples
        """
    
    def sample_spikes(self, n:int=1) -> np.ndarray:
        """
        Sample n data points from the spike model. 
        Parameters:
        -----------
        n : int
            number of samples
        Returns:
        --------
        x : array (n, dim)
            input data
        """
        # latent variables
        gs = self._sample_latents(n)
        # noise
        xs = self._rng.normal(size=(n, self.dim))
        # add the spike
        xs += np.sqrt(self.snr) * (gs[:, None] * self.spike)
        # whiten
        if self.whiten:
            xs = xs @ self._S

        return xs       


    def sample(self, n:int=1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample n data points from the spike model. Probability p_spike of sampling from the spike (y = +1). Otherwise pure Gaussian (y=-1).
        Parameters:
        -----------
        n : int
            number of samples
        Returns:
        --------
        x : array (n, dim)
            input data
        y : array (n,)
            output data
        """
        # Sample Gaussian data and spike data
        x_gauss = self._rng.normal(size=(n, self.dim))
        x_spike = self.sample_spikes(n)
        # Decide which samples are spikes
        spike_mask = self._rng.random(n) < self.p_spike
        # Sample based on mask
        x = np.where(spike_mask[:, None], x_spike, x_gauss)
        y = np.where(spike_mask, 1.0, -1.0)
        return x, y



class RademacherCumulant(SpikedCumulant):
    """
    Spiked cumulant model with arbitrary latent variable.
    """

    def __init__(self, dim , spike:Optional[np.ndarray]=None, snr:float=5.0, rng:Optional[np.random.Generator]=None, whiten:bool=True, p_spike:float=0.5):
        super().__init__(dim, spike , snr,  rng, whiten,p_spike)

    def _sample_latents(self, n:int) -> np.ndarray:
        gs = self._rng.choice([-1, 1], size=n)
        return gs


class SkewedCumulant(SpikedCumulant):
    """
    Skewed cumulant model with Tsodyk's distribution over the latents
    """

    _rho = 0.1  # skewness parameter
    _S = None  # whitening matrix

    def __init__(self, dim , spike:Optional[np.ndarray]=None, snr:float=5.0, rng:Optional[np.random.Generator]=None, whiten:bool=True, p_spike:float=0.5,rho:float=0.7):
        
        """
        Additional parameters:
        -----------
        rho : float
            skewness parameter of Tsodyk's distribution
        """
        super().__init__(dim, spike , snr,  rng, whiten,p_spike)

        self._rho = rho

    def _sample_latents(self, n:int) -> np.ndarray:
        # sample from Tsodyk's prior
        gs = -self._rho * np.ones(n)
        zs = self._rng.uniform(size=n)  # some auxiliary randomness
        gs[zs > (1 - self._rho)] = 1 - self._rho

        # set variance to one
        gs /= np.sqrt(self._rho - self._rho**2)
        return gs


##########################################################
# Generic data generator
##########################################################


class DataGenerator:
    """
    Wraps an InputSampler to deliver samples controlling batch, walkers, repetitions, etc.
    """
    def __init__(self, data_sampler:DataSampler, N_walkers:Optional[int]=1, mode:str='online',dataset:Optional[list[list[Tuple[np.ndarray,np.ndarray]]]]=None, dataset_size:Optional[int]=None, p_repeat:Optional[float]=None):
        """
        Parameters:
        ----------- 
        data_sampler : InputSampler
            instance of InputSampler to generate data from
        N_walkers : int or None
            number of parallel data streams (walkers). If None or 1, single chain mode is used.
        mode : str
            'online' for a fresh sample each time, 'repeat' for including repetitions from dataset (must provide dataset or dataset_size)  
        dataset : list of lists of tuples [[(x, y)]], optional
            Pre-generated dataset to use in 'repeat' mode. If None, will generate a new dataset of size dataset_size.
        dataset_size : int or None, optional
            Size of dataset to generate if dataset is None and mode is 'repeat'.
        p_repeat : float or None, optional
            Probability of sampling from dataset in 'repeat' mode. If None, defaults to 0.0.
        """

        
        if mode not in ['online','repeat']:
            raise ValueError("mode must be 'online' or 'repeat'")
        if dataset is not None and mode=='online':
            print("Warning: dataset provided but mode is 'online'. Dataset will be ignored.")
        if mode=='repeat' and dataset is None and dataset_size is None:
            raise ValueError("For mode 'repeat', must provide either dataset or dataset_size.")
        if not isinstance(data_sampler, DataSampler):
            raise ValueError("sampler must be an instance of InputSampler.")
        
        if dataset is not None and N_walkers != len(dataset):
            raise ValueError("N_walkers must match the length of the provided dataset.")

        if dataset is not None and dataset_size is not None:
            if len(dataset[0]) != dataset_size:
                raise ValueError("dataset_size does not match the size of the provided dataset.")
        if p_repeat is not None and mode != 'repeat':
            print("Warning: p_repeat provided but mode is not 'repeat'. p_repeat will be ignored.")
        if p_repeat is not None:
            if p_repeat < 0.0 or p_repeat > 1.0:
                raise ValueError("p_repeat must be between 0 and 1.")
            
        if N_walkers is None or N_walkers < 1:
            self.N_walkers = 1
        else:
            self.N_walkers = N_walkers
        if mode == 'repeat' and p_repeat is None:
            self.p_repeat = 0.0  # default probability of repetition
        else:
            self.p_repeat = p_repeat

        self.sampler = data_sampler
        if mode=='repeat':
            if dataset is None:
                self.dataset = self.sample_dataset(dataset_size)
            else:
                self.dataset = dataset
        else:
            self.dataset = None
        
        
        if mode == 'repeat':
            self.generator = self.sample_with_repetitions
        else:
            self.generator = self.sample_online


        self.dim = data_sampler.dim
        self.mode = mode
        self.dataset_size = dataset_size
        self.rng = self.sampler._rng
        self._count = 0 # sample counter
        self._state = None  # placeholder for future state tracking


    def sample_online(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate a sample (x, y) for N_walkers in parallel using the sampler.

        Returns:
        --------
        x : array (N_walkers, dim)
            input vector
        y : array (N_walkers,)
            labels
        flag : array (N_walkers,)
            all zeros, indicating fresh samples
        """
        x , y = self.sampler.sample(n=1)
        return x , y , np.zeros_like(y)
    
    def sample_dataset(self,dataset_size:int) -> list[list[Tuple[np.ndarray,np.ndarray]]]:
        """
        Generate a dataset with (x , y) pairs using the sampler.
        Parameters:
        -----------
        dataset_size : int
            Number of samples to generate.
        Returns:
        --------
        dataset : list of lists of tuples [[(x, y)]]
            A list where each element corresponds to a walker and contains a list of tuples with input vectors x and their corresponding label values y.     
        """
        dataset = []
        for _ in range(self.N_walkers):
            data_w = []
            for _ in range(dataset_size):
                x , y = self.sampler.sample(n=1)
                data_w.append((x.flatten(),y.flatten()))
            dataset.append(data_w)
        return dataset
    

    def sample_with_repetitions(self):
        """
        Generate data for N_walkers in parallel, each with probability p_repeat of picking from dataset. Used in 'repeat' mode.
        Returns:
        --------
        x : (N_walkers, dim) array
            Input data for each walker.
        y : (N_walkers,) array
            Target values for each walker.
        flag : (N_walkers,) array
            1 if sampled from dataset, 0 if fresh.
        """
        if self.dataset is None or self.p_repeat is None:
            raise ValueError("No fixed dataset available. Please initialize with dataset_size and p_repeat.")

        repeat_mask = self.rng.random(self.N_walkers) < self.p_repeat
        repeat_indices = self.rng.integers(0, self.dataset_size, size=self.N_walkers)
        # repeat_indices = self.rng.randint(self.dataset_size, size=self.N_walkers)

        data_repeat_x = np.array([self.dataset[w][repeat_indices[w]][0] for w in range(self.N_walkers)])
        data_repeat_y = np.array([self.dataset[w][repeat_indices[w]][1] for w in range(self.N_walkers)])[:,0]

        data_fresh_x , data_fresh_y = self.sampler.sample(n=self.N_walkers) 
        
        x = np.where(repeat_mask[:, None], data_repeat_x, data_fresh_x)
        y = np.where(repeat_mask, data_repeat_y, data_fresh_y)
        flag = repeat_mask.astype(int)

        return x, y, flag
    

    ## Variations for the distribution of samples

    def sample_twice(self):
        """
        Use sample_with_repetitions to get the same sample twice in a row.
        Returns:
        --------
        x : (N_walkers, dim) array
            Input data for each walker.
        y : (N_walkers,) array
            Target values for each walker.
        flag : (N_walkers,) array
            1 if sampled from dataset, 0 if fresh.
        """
        if self._count == 0:
            self._state = self.generator()
            self._count = 1
            return self._state
        else:
            self._count = 0
            return self._state




    def get_dataset(self, how='arrays'):
        """
        Get the fixed dataset if it exists.
        Parameters:
        -----------
        how : str
            'tuple' to return as list of tuples [(x, y)], 'array' to return as two numpy arrays (X, Y).
        Returns:
        --------
        dataset : list of tuples [(x, y)] or (X, Y) arrays
            The fixed dataset in the requested format.
        """

        if self.dataset is None:
            raise ValueError("No fixed dataset available. Please initialize with dataset_size and p_repeat.")
        
        if self.N_walkers is not None and self.N_walkers > 1 and how == 'arrays':
            X = np.array([[x for x, y in walker_data] for walker_data in self.dataset])
            Y = np.array([[y for x, y in walker_data] for walker_data in self.dataset])
            return X, Y
        
        elif (self.N_walkers is None or self.N_walkers == 1) and how == 'arrays':
            X = np.array([x for x, y in self.dataset])
            Y = np.array([y for x, y in self.dataset])
            return X, Y
        elif how == 'tuple':
            return self.dataset
        else:
            raise ValueError("Invalid 'how' parameter. Use 'tuple' or 'arrays'.")


# ##########################################################
# Old data generator classes (to be removed)
##########################################################

# import numpy as np
# from .functions import activations


# class SpikeGenerator:
#     def __init__(self, d, u_spike, snr, f_spike = 0.5, N_walkers = None, dataset_size=None, p_repeat=None, mode='online'):
#         """
#         Initialize the data generator.
#         Parameters:
#         -----------
#         d : int
#             Dimensionality of the input data.
#         u_spike : (d,) array
#             Spike direction of the data model.
#         snr : float (0 < snr < 1)
#             Signal to noise ratio.
#         f_spike: float
#             Fraction of spiked data samples (or probability of sampling a spike during online).
#         N_walkers : int or None
#             If provided, number of parallel data streams (walkers) to generate data for.
#         dataset_size : int or None
#             If provided, the size of the fixed dataset to sample from.
#         p_repeat : float or None
#             Probability of sampling from the fixed dataset instead of generating new data.
#         mode : str
#             'online' for fresh data each time, 'repeat' for sampling from a fixed dataset (must provide dataset_size and p_repeat also).
#         Raises:
#         -------
#         ValueError : if u_spike is not a numpy array of shape (d,)
#         ValueError : if dataset_size is provided but p_repeat is not, or vice versa
#         ValueError : if p_repeat is not in [0, 1]
#         """
#         #
#         if not isinstance(u_spike, np.ndarray) or u_spike.shape != (d,):
#             raise ValueError(f"w_teacher must be a numpy array of shape ({d},)")
#         if (dataset_size is None) != (p_repeat is None):
#             raise ValueError("Both dataset_size and p_repeat must be provided together or both be None.")
#         if p_repeat is not None and not (0 <= p_repeat <= 1):
#             raise ValueError("p_repeat must be in the range [0, 1].")
#         if mode not in ['online', 'repeat']:
#             raise ValueError("mode must be either 'online' or 'repeat'.")
#         if mode == 'repeat' and (dataset_size is None or p_repeat is None):
#             raise ValueError("For 'repeat' mode, both dataset_size and p_repeat must be provided.")
#         if not (0 <= snr <= 1):
#             raise ValueError(f"SNR must be between 0 and 1, got {snr}")

        
#         # Initialize attributes
#         self.d = d
#         self.N_walkers = N_walkers
#         self.u_spike = u_spike
#         self.dataset_size = dataset_size
#         self.p_repeat = p_repeat
#         self.snr = snr
#         self.f_spike = f_spike

#         # Create fixed dataset if needed
#         if self.N_walkers is None or self.N_walkers == 1:
#             if dataset_size is not None and mode == 'repeat':
#                 self.dataset = self.generate_batch_spikes(dataset_size)
#             else:
#                 self.dataset = None
#         else:
#             if dataset_size is not None and mode == 'repeat':
#                 self.dataset = self.generate_batch_spikes_walkers(dataset_size)
#             else:
#                 self.dataset = None

#         # Define the data generation method
#         if self.N_walkers is None or self.N_walkers == 1:
#             if mode == 'online':
#                 self.generate = self.generate_spike
#             else:  # mode == 'repeat'
#                 self.generate = self.generate_repeating_spikes
#         else:  # Multiple walkers
#             if mode == 'online':
#                 self.generate = self.generate_spike_walkers
#             else:  # mode == 'repeat'
#                 self.generate = self.generate_repeating_spike_walkers

#     def generate_spike(self):
#         """
#         Generate a single (x, y) pair where x is drawn from the spike cumulant model and y the label (+1,-1).
#         Returns:
#         --------
#         x : (d,) array
#             Input data.
#         y : float
#             Target value.
#         """
#         x = np.random.normal(size=self.d)
#         if np.random.random() > self.f_spike:
#             y = -1
#         else:
#             y = +1
#             g = np.random.choice([-1.,+1.])
#             whitening_term = self.u_spike * np.dot(self.u_spike,x)
#             x += self.snr * g * self.u_spike + (np.sqrt(1 - self.snr**2) - 1) * whitening_term

#         return x, y , 0   

#     def generate_batch_spikes(self, N):
#         """
#         Generate a batch of (x, y) pairs using Gaussian inputs.

#         Parameters:
#         -----------
#         N : int
#             Number of samples to generate.

#         Returns:
#         --------
#         dataset : list of tuples [(x, y)]
#             A list where each element is a tuple containing an input vector x and its corresponding target value y.
#         """
#         dataset = []
#         for _ in range(N):
#             dataset.append((self.generate_spike()[:2]))
#         return dataset

#     def generate_repeating_spikes(self):
#         """
#         Generate a single (x, y) pair by sampling with probability p_repeat from a fixed dataset
#         of spiked cumulant data and with probability 1 - p_repeat get a fresh sample.
#         Returns:
#         --------
#         x : (d,) array
#             Input data.
#         y : float
#             Target value.
#         """
#         flag = 0
#         if self.dataset is not None and np.random.rand() < self.p_repeat:
#             # Sample from the fixed dataset
#             idx = np.random.randint(self.dataset_size)
#             x, y = self.dataset[idx]
#             flag = 1
#         else:
#             # Generate fresh data
#             x , y , flag = self.generate_spike()
#         return x, y , flag


#     def generate_spike_walkers(self):
#         """
#         Generate fresh spiked data for N_walkers in parallel.
#         Returns:
#         --------
#         x : (N_walkers, d) array
#             Input data for each walker.
#         y : (N_walkers,) array
#             Target values for each walker.
#         flag : (N_walkers,) array
#             0 for all, since all are fresh samples.
#         """
#         x = np.random.normal(size=(self.N_walkers, self.d))
#         g = np.random.choice([1,-1],p=[0.5,0.5],size=self.N_walkers)
#         spike_term = np.outer(g,self.u_spike)
#         whitening_term =  (x @ self.u_spike[:,None]) @ self.u_spike[None,:] 
#         y = np.random.choice([1,-1],p=[self.f_spike,1-self.f_spike],size=self.N_walkers)
#         mask = 0.5*(y + 1)
#         x += mask[:,None] * (self.snr*spike_term + (np.sqrt(1 - self.snr**2) - 1) * whitening_term)
#         flag = np.zeros(self.N_walkers, dtype=int)
#         return x, y, flag
    
#     def generate_batch_spikes_walkers(self, N):
#         """
#         Generate a batch of (x, y) pairs for each walker using spike model.

#         Parameters:
#         -----------
#         N : int
#             Number of samples to generate for each walker.

#         Returns:
#         --------
#         datasets : list of lists of tuples [[(x, y)]]
#             A list where each element corresponds to a walker and contains a list of tuples with input vectors x and their corresponding target values y.
#         """
#         datasets = []
#         for _ in range(self.N_walkers):
#             datasets.append((self.generate_batch_spikes(N)))
#         return datasets

#     def generate_repeating_spike_walkers(self):
#         """
#         Generate data for N_walkers in parallel, each with probability p_repeat of picking from its dataset.
#         Returns:
#         --------
#         x : (N_walkers, d) array
#             Input data for each walker.
#         y : (N_walkers,) array
#             Target values for each walker.
#         flag : (N_walkers,) array
#             1 if sampled from dataset, 0 if fresh.
#         """
#         if self.dataset is None:
#             raise ValueError("No fixed dataset available. Please initialize with dataset_size and p_repeat.")

#         repeat_mask = np.random.rand(self.N_walkers) < self.p_repeat
#         repeat_indices = np.random.randint(self.dataset_size, size=self.N_walkers)

#         data_repeat_x = np.array([self.dataset[w][repeat_indices[w]][0] for w in range(self.N_walkers)])
#         data_repeat_y = np.array([self.dataset[w][repeat_indices[w]][1] for w in range(self.N_walkers)])
        
#         data_fresh_x , data_fresh_y = self.generate_spike_walkers()[:2] 

#         x = np.where(repeat_mask[:, None], data_repeat_x, data_fresh_x)
#         y = np.where(repeat_mask, data_repeat_y, data_fresh_y)
#         flag = repeat_mask.astype(int)
#         return x, y, flag


#     def get_dataset(self, how='arrays'):
#         """
#         Get the fixed dataset if it exists.
#         Parameters:
#         -----------
#         how : str
#             'tuple' to return as list of tuples [(x, y)], 'array' to return as two numpy arrays (X, Y).
#         Returns:
#         --------
#         dataset : list of tuples [(x, y)] or (X, Y) arrays
#             The fixed dataset in the requested format.
#         """

#         if self.dataset is None:
#             raise ValueError("No fixed dataset available. Please initialize with dataset_size and p_repeat.")
        
#         if self.N_walkers is not None and self.N_walkers > 1 and how == 'arrays':
#             X = np.array([[x for x, y in walker_data] for walker_data in self.dataset])
#             Y = np.array([[y for x, y in walker_data] for walker_data in self.dataset])
#             return X, Y
        
#         elif (self.N_walkers is None or self.N_walkers == 1) and how == 'arrays':
#             X = np.array([x for x, y in self.dataset])
#             Y = np.array([y for x, y in self.dataset])
#             return X, Y
#         elif how == 'tuple':
#             return self.dataset
#         else:
#             raise ValueError("Invalid 'how' parameter. Use 'tuple' or 'arrays'.")



# class DataGenerator:
#     def __init__(self, d, teacher_fun, w_teacher, noise=0.0, N_walkers = None, dataset_size=None, p_repeat=None, mode='online'):
#         """
#         Initialize the data generator.
#         Parameters:
#         -----------
#         d : int
#             Dimensionality of the input data.
#         teacher_fun : callable  
#             Activation function for the teacher model.
#         w_teacher : (d,) array
#             Weights of the teacher model.
#         noise : float
#             Standard deviation of the Gaussian noise added to the outputs.
#         N_walkers : int or None
#             If provided, number of parallel data streams (walkers) to generate data for.
#         dataset_size : int or None
#             If provided, the size of the fixed dataset to sample from.
#         p_repeat : float or None
#             Probability of sampling from the fixed dataset instead of generating new data.
#         mode : str
#             'online' for fresh data each time, 'repeat' for sampling from a fixed dataset (must provide dataset_size and p_repeat also).
#         Raises:
#         -------
#         ValueError : if w_teacher is not a numpy array of shape (d,)
#         ValueError : if dataset_size is provided but p_repeat is not, or vice versa
#         ValueError : if p_repeat is not in [0, 1]
#         """
#         #
#         if not isinstance(w_teacher, np.ndarray) or w_teacher.shape != (d,):
#             raise ValueError(f"w_teacher must be a numpy array of shape ({d},)")
#         if (dataset_size is None) != (p_repeat is None):
#             raise ValueError("Both dataset_size and p_repeat must be provided together or both be None.")
#         if p_repeat is not None and not (0 <= p_repeat <= 1):
#             raise ValueError("p_repeat must be in the range [0, 1].")
#         if mode not in ['online', 'repeat']:
#             raise ValueError("mode must be either 'online' or 'repeat'.")
#         if mode == 'repeat' and (dataset_size is None or p_repeat is None):
#             raise ValueError("For 'repeat' mode, both dataset_size and p_repeat must be provided.")
        
#         # Initialize attributes
#         self.d = d
#         self.N_walkers = N_walkers
#         self.teacher_fun = activations[teacher_fun][0]
#         self.w_teacher = w_teacher
#         self.noise = noise
#         self.dataset_size = dataset_size
#         self.p_repeat = p_repeat

#         # Create fixed dataset if needed
#         if self.N_walkers is None or self.N_walkers == 1:
#             if dataset_size is not None and mode == 'repeat':
#                 self.dataset = self.generate_batch(dataset_size)
#             else:
#                 self.dataset = None
#         else:
#             if dataset_size is not None and mode == 'repeat':
#                 self.dataset = self.generate_batch_walkers(dataset_size)
#             else:
#                 self.dataset = None

#         # Define the data generation method
#         if self.N_walkers is None or self.N_walkers == 1:
#             if mode == 'online':
#                 self.generate = self.generate_gaussian
#             else:  # mode == 'repeat'
#                 self.generate = self.generate_repeating
#         else:  # Multiple walkers
#             if mode == 'online':
#                 self.generate = self.generate_gaussian_walkers
#             else:  # mode == 'repeat'
#                 self.generate = self.generate_repeating_walkers

    
#     def generate_gaussian(self):
#         """
#         Generate a single (x, y) pair where x is drawn from a Gaussian distribution and y is generated using the teacher function.
#         Returns:
#         --------
#         x : (d,) array
#             Input data.
#         y : float
#             Target value.
#         """
#         x = np.random.normal(size=self.d)
#         y = self.teacher_fun(np.dot(self.w_teacher, x)) + self.noise * np.random.randn()
#         return x, y , 0

#     def generate_batch(self, N):
#         """
#         Generate a batch of (x, y) pairs using Gaussian inputs.

#         Parameters:
#         -----------
#         N : int
#             Number of samples to generate.

#         Returns:
#         --------
#         dataset : list of tuples [(x, y)]
#             A list where each element is a tuple containing an input vector x and its corresponding target value y.
#         """
#         dataset = []
#         for _ in range(N):
#             dataset.append((self.generate_gaussian()[:2]))
#         return dataset
        
#     def generate_repeating(self):
#         """
#         Generate a single (x, y) pair by sampling with probability p_repeat from a fixed dataset
#         and with probability 1 - p_repeat get a fresh sample.
#         Returns:
#         --------
#         x : (d,) array
#             Input data.
#         y : float
#             Target value.
#         """
#         flag = 0
#         if self.dataset is not None and np.random.rand() < self.p_repeat:
#             # Sample from the fixed dataset
#             idx = np.random.randint(self.dataset_size)
#             x, y = self.dataset[idx]
#             flag = 1
#         else:
#             # Generate fresh data
#             x = np.random.normal(size=self.d)
#             y = self.teacher_fun(np.dot(self.w_teacher, x)) + self.noise * np.random.randn()
#         return x, y , flag

#     def generate_gaussian_walkers(self):
#         """
#         Generate fresh Gaussian data for N_walkers in parallel.
#         Returns:
#         --------
#         x : (N_walkers, d) array
#             Input data for each walker.
#         y : (N_walkers,) array
#             Target values for each walker.
#         flag : (N_walkers,) array
#             0 for all, since all are fresh samples.
#         """
#         x = np.random.normal(size=(self.N_walkers, self.d))
#         y = self.teacher_fun(np.dot(x, self.w_teacher)) + self.noise * np.random.randn(self.N_walkers)
#         flag = np.zeros(self.N_walkers, dtype=int)
#         return x, y, flag

    
#     def generate_batch_walkers(self, N):
#         """
#         Generate a batch of (x, y) pairs for each walker using Gaussian inputs.

#         Parameters:
#         -----------
#         N : int
#             Number of samples to generate for each walker.

#         Returns:
#         --------
#         datasets : list of lists of tuples [[(x, y)]]
#             A list where each element corresponds to a walker and contains a list of tuples with input vectors x and their corresponding target values y.
#         """
#         datasets = []
#         for _ in range(self.N_walkers):
#             datasets.append((self.generate_batch(N)))
#         return datasets

#     def generate_repeating_walkers(self):
#         """
#         Generate data for N_walkers in parallel, each with probability p_repeat of picking from its dataset.
#         Returns:
#         --------
#         x : (N_walkers, d) array
#             Input data for each walker.
#         y : (N_walkers,) array
#             Target values for each walker.
#         flag : (N_walkers,) array
#             1 if sampled from dataset, 0 if fresh.
#         """
#         if self.dataset is None:
#             raise ValueError("No fixed dataset available. Please initialize with dataset_size and p_repeat.")

#         repeat_mask = np.random.rand(self.N_walkers) < self.p_repeat
#         repeat_indices = np.random.randint(self.dataset_size, size=self.N_walkers)

#         data_repeat_x = np.array([self.dataset[w][repeat_indices[w]][0] for w in range(self.N_walkers)])
#         data_repeat_y = np.array([self.dataset[w][repeat_indices[w]][1] for w in range(self.N_walkers)])
        
#         data_fresh_x , data_fresh_y = self.generate_gaussian_walkers()[:2] 

#         x = np.where(repeat_mask[:, None], data_repeat_x, data_fresh_x)
#         y = np.where(repeat_mask, data_repeat_y, data_fresh_y)
#         flag = repeat_mask.astype(int)
#         return x, y, flag


#     def get_dataset(self, how='arrays'):
#         """
#         Get the fixed dataset if it exists.
#         Parameters:
#         -----------
#         how : str
#             'tuple' to return as list of tuples [(x, y)], 'array' to return as two numpy arrays (X, Y).
#         Returns:
#         --------
#         dataset : list of tuples [(x, y)] or (X, Y) arrays
#             The fixed dataset in the requested format.
#         """

#         if self.dataset is None:
#             raise ValueError("No fixed dataset available. Please initialize with dataset_size and p_repeat.")
        
#         if self.N_walkers is not None and self.N_walkers > 1 and how == 'arrays':
#             X = np.array([[x for x, y in walker_data] for walker_data in self.dataset])
#             Y = np.array([[y for x, y in walker_data] for walker_data in self.dataset])
#             return X, Y
        
#         elif (self.N_walkers is None or self.N_walkers == 1) and how == 'arrays':
#             X = np.array([x for x, y in self.dataset])
#             Y = np.array([y for x, y in self.dataset])
#             return X, Y
#         elif how == 'tuple':
#             return self.dataset
#         else:
#             raise ValueError("Invalid 'how' parameter. Use 'tuple' or 'arrays'.")

