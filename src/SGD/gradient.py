"""
This module contains various gradient steps.
"""

import numpy as np
import numba as nb


# @nb.njit
def online_grad_mse(w,x,y,fun,deriv,spherical=True):
    '''
    Compute the gradient of the MSE loss function for a single point.
    
    w : (d,) array, weights for each run
    x : (d,)  array, input data for each run
    y : scalar, target value for each run
    fun : callable, function to compute predictions
    deriv : callable, derivative of the function fun
    spherical : bool, if true project out the radial component of gradient
    
    Returns:
    grad : (d,) array, gradients for each run
    '''
    preac = np.dot(w , x)  # scalar
    pred = fun(preac)  # scalar
    err = y - pred  # scalar
    grad = - (err * deriv(preac))* x   # (d,)
    if spherical: grad -= np.dot(w,grad) * w
    return grad

def online_grad_mse_walkers(w,x,y,fun,deriv,spherical=False):
    '''
    Compute the gradient of the MSE loss function for a single point
    across many independent walkers.
    
    w : (N_walkers,d) array, weights for each run
    x : (N_walkers,d)  array, input data for each run
    y : (N_walkers) array, target values for each run
    fun : callable, function to compute predictions
    deriv : callable, derivative of the function fun
    spherical : bool, if true project out the radial component of gradient
           
    Returns:
    grad : (N_walkers,d) array, gradients for each run
    '''

    preac = np.sum(w * x, axis=-1)  # (N_walkers)
    pred = fun(preac)  # (N_walkers)
    err = y - pred  # (N_walkers)
    grad = - (err * deriv(preac))[:,None] * x   # (N_walkers,d)
    if spherical:
        dot = np.sum(w * grad, axis=1) # (N_walkers)
        grad -= dot[:,None] * w
    return grad  # (N_walkers,d)


def online_grad_corr_runs(w,x,y,fun,deriv,spherical=False):
    '''
    Compute the gradient of the Correlation loss function for a single point
    across many independent runs.
    
    w : (N_runs,d) array, weights for each run
    x : (N_runs,d)  array, input data for each run
    y : (N_runs) array, target values for each run
    fun : callable, function to compute predictions
    deriv : callable, derivative of the function fun
    spherical : bool, if true project out the radial component of gradient
    
    Returns:
    grad : (N_runs,d) array, gradients for each run
    '''

    preac = np.sum(w * x, axis=-1)  # (N_runs)
    grad = - (y * deriv(preac))[:,None] * x   # (N_runs,d)
    if spherical:
        dot = np.sum(w * grad, axis=1) # (N_runs)
        grad -= dot[:,None] * w
    return grad  # (N_runs,d)

def online_grad_corr(w,x,y,fun,deriv,spherical=False):
    '''
    Compute the gradient of the Correlation loss function for a single point.
    
    w : (d,) array, weights for each run
    x : (d,)  array, input data for each run
    y : scalar, target value for each run
    fun : callable, function to compute predictions
    deriv : callable, derivative of the function fun
    spherical: bool, whether to project out the radial component of grad or not
    
    Returns:
    grad : (d,) array, gradients for each run
    '''
    preac = w @ x  # scalar
    grad = - (y * deriv(preac)) * x  # (d,)
    if spherical: grad -= np.dot(w,grad) * w
    return grad



# Dictionary to map function names to their implementations
gradients = {
    "mse": {
        'single_chain': online_grad_mse,
        'multi_chain': online_grad_mse_walkers
    },
    "corr": {
        'single_chain': online_grad_corr,
        'multi_chain': online_grad_corr_runs
    }
}
