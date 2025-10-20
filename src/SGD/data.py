"""
This module contains a class for data generation .
"""

import numpy as np
from .functions import activations


class SpikeGenerator:
    def __init__(self, d, u_spike, snr, f_spike = 0.5, N_walkers = None, dataset_size=None, p_repeat=None, mode='online'):
        """
        Initialize the data generator.
        Parameters:
        -----------
        d : int
            Dimensionality of the input data.
        u_spike : (d,) array
            Spike direction of the data model.
        snr : float (0 < snr < 1)
            Signal to noise ratio.
        f_spike: float
            Fraction of spiked data samples (or probability of sampling a spike during online).
        N_walkers : int or None
            If provided, number of parallel data streams (walkers) to generate data for.
        dataset_size : int or None
            If provided, the size of the fixed dataset to sample from.
        p_repeat : float or None
            Probability of sampling from the fixed dataset instead of generating new data.
        mode : str
            'online' for fresh data each time, 'repeat' for sampling from a fixed dataset (must provide dataset_size and p_repeat also).
        Raises:
        -------
        ValueError : if u_spike is not a numpy array of shape (d,)
        ValueError : if dataset_size is provided but p_repeat is not, or vice versa
        ValueError : if p_repeat is not in [0, 1]
        """
        #
        if not isinstance(u_spike, np.ndarray) or u_spike.shape != (d,):
            raise ValueError(f"w_teacher must be a numpy array of shape ({d},)")
        if (dataset_size is None) != (p_repeat is None):
            raise ValueError("Both dataset_size and p_repeat must be provided together or both be None.")
        if p_repeat is not None and not (0 <= p_repeat <= 1):
            raise ValueError("p_repeat must be in the range [0, 1].")
        if mode not in ['online', 'repeat']:
            raise ValueError("mode must be either 'online' or 'repeat'.")
        if mode == 'repeat' and (dataset_size is None or p_repeat is None):
            raise ValueError("For 'repeat' mode, both dataset_size and p_repeat must be provided.")
        if not (0 <= snr <= 1):
            raise ValueError(f"SNR must be between 0 and 1, got {snr}")

        
        # Initialize attributes
        self.d = d
        self.N_walkers = N_walkers
        self.u_spike = u_spike
        self.dataset_size = dataset_size
        self.p_repeat = p_repeat
        self.snr = snr
        self.f_spike = f_spike

        # Create fixed dataset if needed
        if self.N_walkers is None or self.N_walkers == 1:
            if dataset_size is not None and mode == 'repeat':
                self.dataset = self.generate_batch_spikes(dataset_size)
            else:
                self.dataset = None
        else:
            if dataset_size is not None and mode == 'repeat':
                self.dataset = self.generate_batch_spikes_walkers(dataset_size)
            else:
                self.dataset = None

        # Define the data generation method
        if self.N_walkers is None or self.N_walkers == 1:
            if mode == 'online':
                self.generate = self.generate_spike
            else:  # mode == 'repeat'
                self.generate = self.generate_repeating_spikes
        else:  # Multiple walkers
            if mode == 'online':
                self.generate = self.generate_spike_walkers
            else:  # mode == 'repeat'
                self.generate = self.generate_repeating_spike_walkers

    def generate_spike(self):
        """
        Generate a single (x, y) pair where x is drawn from the spike cumulant model and y the label (+1,-1).
        Returns:
        --------
        x : (d,) array
            Input data.
        y : float
            Target value.
        """
        x = np.random.normal(size=self.d)
        if np.random.random() > self.f_spike:
            y = -1
        else:
            y = +1
            g = np.random.choice([-1.,+1.])
            whitening_term = self.u_spike * np.dot(self.u_spike,x)
            x += self.snr * g * self.u_spike + (np.sqrt(1 - self.snr**2) - 1) * whitening_term

        return x, y , 0   

    def generate_batch_spikes(self, N):
        """
        Generate a batch of (x, y) pairs using Gaussian inputs.

        Parameters:
        -----------
        N : int
            Number of samples to generate.

        Returns:
        --------
        dataset : list of tuples [(x, y)]
            A list where each element is a tuple containing an input vector x and its corresponding target value y.
        """
        dataset = []
        for _ in range(N):
            dataset.append((self.generate_spike()[:2]))
        return dataset

    def generate_repeating_spikes(self):
        """
        Generate a single (x, y) pair by sampling with probability p_repeat from a fixed dataset
        of spiked cumulant data and with probability 1 - p_repeat get a fresh sample.
        Returns:
        --------
        x : (d,) array
            Input data.
        y : float
            Target value.
        """
        flag = 0
        if self.dataset is not None and np.random.rand() < self.p_repeat:
            # Sample from the fixed dataset
            idx = np.random.randint(self.dataset_size)
            x, y = self.dataset[idx]
            flag = 1
        else:
            # Generate fresh data
            x , y , flag = self.generate_spike()
        return x, y , flag


    def generate_spike_walkers(self):
        """
        Generate fresh spiked data for N_walkers in parallel.
        Returns:
        --------
        x : (N_walkers, d) array
            Input data for each walker.
        y : (N_walkers,) array
            Target values for each walker.
        flag : (N_walkers,) array
            0 for all, since all are fresh samples.
        """
        x = np.random.normal(size=(self.N_walkers, self.d))
        g = np.random.choice([1,-1],p=[0.5,0.5],size=self.N_walkers)
        spike_term = np.outer(g,self.u_spike)
        whitening_term =  (x @ self.u_spike[:,None]) @ self.u_spike[None,:] 
        y = np.random.choice([1,-1],p=[self.f_spike,1-self.f_spike],size=self.N_walkers)
        mask = 0.5*(y + 1)
        x += mask[:,None] * (self.snr*spike_term + (np.sqrt(1 - self.snr**2) - 1) * whitening_term)
        flag = np.zeros(self.N_walkers, dtype=int)
        return x, y, flag
    
    def generate_batch_spikes_walkers(self, N):
        """
        Generate a batch of (x, y) pairs for each walker using spike model.

        Parameters:
        -----------
        N : int
            Number of samples to generate for each walker.

        Returns:
        --------
        datasets : list of lists of tuples [[(x, y)]]
            A list where each element corresponds to a walker and contains a list of tuples with input vectors x and their corresponding target values y.
        """
        datasets = []
        for _ in range(self.N_walkers):
            datasets.append((self.generate_batch_spikes(N)))
        return datasets

    def generate_repeating_spike_walkers(self):
        """
        Generate data for N_walkers in parallel, each with probability p_repeat of picking from its dataset.
        Returns:
        --------
        x : (N_walkers, d) array
            Input data for each walker.
        y : (N_walkers,) array
            Target values for each walker.
        flag : (N_walkers,) array
            1 if sampled from dataset, 0 if fresh.
        """
        if self.dataset is None:
            raise ValueError("No fixed dataset available. Please initialize with dataset_size and p_repeat.")

        repeat_mask = np.random.rand(self.N_walkers) < self.p_repeat
        repeat_indices = np.random.randint(self.dataset_size, size=self.N_walkers)

        data_repeat_x = np.array([self.dataset[w][repeat_indices[w]][0] for w in range(self.N_walkers)])
        data_repeat_y = np.array([self.dataset[w][repeat_indices[w]][1] for w in range(self.N_walkers)])
        
        data_fresh_x , data_fresh_y = self.generate_spike_walkers()[:2] 

        x = np.where(repeat_mask[:, None], data_repeat_x, data_fresh_x)
        y = np.where(repeat_mask, data_repeat_y, data_fresh_y)
        flag = repeat_mask.astype(int)
        return x, y, flag


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



class DataGenerator:
    def __init__(self, d, teacher_fun, w_teacher, noise=0.0, N_walkers = None, dataset_size=None, p_repeat=None, mode='online'):
        """
        Initialize the data generator.
        Parameters:
        -----------
        d : int
            Dimensionality of the input data.
        teacher_fun : callable  
            Activation function for the teacher model.
        w_teacher : (d,) array
            Weights of the teacher model.
        noise : float
            Standard deviation of the Gaussian noise added to the outputs.
        N_walkers : int or None
            If provided, number of parallel data streams (walkers) to generate data for.
        dataset_size : int or None
            If provided, the size of the fixed dataset to sample from.
        p_repeat : float or None
            Probability of sampling from the fixed dataset instead of generating new data.
        mode : str
            'online' for fresh data each time, 'repeat' for sampling from a fixed dataset (must provide dataset_size and p_repeat also).
        Raises:
        -------
        ValueError : if w_teacher is not a numpy array of shape (d,)
        ValueError : if dataset_size is provided but p_repeat is not, or vice versa
        ValueError : if p_repeat is not in [0, 1]
        """
        #
        if not isinstance(w_teacher, np.ndarray) or w_teacher.shape != (d,):
            raise ValueError(f"w_teacher must be a numpy array of shape ({d},)")
        if (dataset_size is None) != (p_repeat is None):
            raise ValueError("Both dataset_size and p_repeat must be provided together or both be None.")
        if p_repeat is not None and not (0 <= p_repeat <= 1):
            raise ValueError("p_repeat must be in the range [0, 1].")
        if mode not in ['online', 'repeat']:
            raise ValueError("mode must be either 'online' or 'repeat'.")
        if mode == 'repeat' and (dataset_size is None or p_repeat is None):
            raise ValueError("For 'repeat' mode, both dataset_size and p_repeat must be provided.")
        
        # Initialize attributes
        self.d = d
        self.N_walkers = N_walkers
        self.teacher_fun = activations[teacher_fun][0]
        self.w_teacher = w_teacher
        self.noise = noise
        self.dataset_size = dataset_size
        self.p_repeat = p_repeat

        # Create fixed dataset if needed
        if self.N_walkers is None or self.N_walkers == 1:
            if dataset_size is not None and mode == 'repeat':
                self.dataset = self.generate_batch(dataset_size)
            else:
                self.dataset = None
        else:
            if dataset_size is not None and mode == 'repeat':
                self.dataset = self.generate_batch_walkers(dataset_size)
            else:
                self.dataset = None

        # Define the data generation method
        if self.N_walkers is None or self.N_walkers == 1:
            if mode == 'online':
                self.generate = self.generate_gaussian
            else:  # mode == 'repeat'
                self.generate = self.generate_repeating
        else:  # Multiple walkers
            if mode == 'online':
                self.generate = self.generate_gaussian_walkers
            else:  # mode == 'repeat'
                self.generate = self.generate_repeating_walkers

    
    def generate_gaussian(self):
        """
        Generate a single (x, y) pair where x is drawn from a Gaussian distribution and y is generated using the teacher function.
        Returns:
        --------
        x : (d,) array
            Input data.
        y : float
            Target value.
        """
        x = np.random.normal(size=self.d)
        y = self.teacher_fun(np.dot(self.w_teacher, x)) + self.noise * np.random.randn()
        return x, y , 0

    def generate_batch(self, N):
        """
        Generate a batch of (x, y) pairs using Gaussian inputs.

        Parameters:
        -----------
        N : int
            Number of samples to generate.

        Returns:
        --------
        dataset : list of tuples [(x, y)]
            A list where each element is a tuple containing an input vector x and its corresponding target value y.
        """
        dataset = []
        for _ in range(N):
            dataset.append((self.generate_gaussian()[:2]))
        return dataset
        
    def generate_repeating(self):
        """
        Generate a single (x, y) pair by sampling with probability p_repeat from a fixed dataset
        and with probability 1 - p_repeat get a fresh sample.
        Returns:
        --------
        x : (d,) array
            Input data.
        y : float
            Target value.
        """
        flag = 0
        if self.dataset is not None and np.random.rand() < self.p_repeat:
            # Sample from the fixed dataset
            idx = np.random.randint(self.dataset_size)
            x, y = self.dataset[idx]
            flag = 1
        else:
            # Generate fresh data
            x = np.random.normal(size=self.d)
            y = self.teacher_fun(np.dot(self.w_teacher, x)) + self.noise * np.random.randn()
        return x, y , flag

    def generate_gaussian_walkers(self):
        """
        Generate fresh Gaussian data for N_walkers in parallel.
        Returns:
        --------
        x : (N_walkers, d) array
            Input data for each walker.
        y : (N_walkers,) array
            Target values for each walker.
        flag : (N_walkers,) array
            0 for all, since all are fresh samples.
        """
        x = np.random.normal(size=(self.N_walkers, self.d))
        y = self.teacher_fun(np.dot(x, self.w_teacher)) + self.noise * np.random.randn(self.N_walkers)
        flag = np.zeros(self.N_walkers, dtype=int)
        return x, y, flag

    
    def generate_batch_walkers(self, N):
        """
        Generate a batch of (x, y) pairs for each walker using Gaussian inputs.

        Parameters:
        -----------
        N : int
            Number of samples to generate for each walker.

        Returns:
        --------
        datasets : list of lists of tuples [[(x, y)]]
            A list where each element corresponds to a walker and contains a list of tuples with input vectors x and their corresponding target values y.
        """
        datasets = []
        for _ in range(self.N_walkers):
            datasets.append((self.generate_batch(N)))
        return datasets

    def generate_repeating_walkers(self):
        """
        Generate data for N_walkers in parallel, each with probability p_repeat of picking from its dataset.
        Returns:
        --------
        x : (N_walkers, d) array
            Input data for each walker.
        y : (N_walkers,) array
            Target values for each walker.
        flag : (N_walkers,) array
            1 if sampled from dataset, 0 if fresh.
        """
        if self.dataset is None:
            raise ValueError("No fixed dataset available. Please initialize with dataset_size and p_repeat.")

        repeat_mask = np.random.rand(self.N_walkers) < self.p_repeat
        repeat_indices = np.random.randint(self.dataset_size, size=self.N_walkers)

        data_repeat_x = np.array([self.dataset[w][repeat_indices[w]][0] for w in range(self.N_walkers)])
        data_repeat_y = np.array([self.dataset[w][repeat_indices[w]][1] for w in range(self.N_walkers)])
        
        data_fresh_x , data_fresh_y = self.generate_gaussian_walkers()[:2] 

        x = np.where(repeat_mask[:, None], data_repeat_x, data_fresh_x)
        y = np.where(repeat_mask, data_repeat_y, data_fresh_y)
        flag = repeat_mask.astype(int)
        return x, y, flag


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

