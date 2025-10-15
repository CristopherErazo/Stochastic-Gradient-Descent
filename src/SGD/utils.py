"""
This module contains various utility function.
"""

import numpy as np
from .functions import activations

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

class _NoOpPBar:
    """A no-operation progress bar that does nothing."""
    def __init__(self, *args, **kwargs):
        pass

    def update(self, count):
        pass

    def close(self):
        pass


def get_progress_bar(enabled, total):
    """
    Return a progress bar if enabled, otherwise return a no-op progress bar.

    Parameters:
    ----------
    - enabled: bool, whether to display the progress bar.
    - total: int, the total number of iterations.

    Returns:
    ----------
    - A progress bar object.
    """
    if enabled and tqdm is not None:
        return tqdm(total=total)
    return _NoOpPBar()

def initialize_weights(d, N_walkers=None, m0=0.0, mode='random'):
    """
    Initialize the teacher and student directions.
    Both directions are sampled in the d-dimensional sphere of radius 1, 
    with the student direction having an overlap of (w* . w) = m0 with the teacher direction.

    Parameters:
    -----------
    d : int
        Dimension of the space.
    N_walkers None int, optional
        Number of walkers (independent students). Default is None.
    m0 : float, optional
        Desired overlap between teacher and student directions. Default is 0.0.
    mode : str, optional
        Initialization mode: 'random', 'fixed', or 'positive'. Default is 'random'.

    Returns:
    --------
    w_star : ndarray
        Teacher direction of shape (d,).
    w0 : ndarray
        Student directions of shape (N_walkers,d) or (d,) if N_walkers=None.
    """
    # Sample teacher direction
    w_star = np.random.normal(size=d)
    w_star /= np.linalg.norm(w_star)

    if N_walkers is None:
        N_walkers = 1  # Default to single walker if None
        
    # Sample student directions at initialization
    w0 = np.random.normal(size=(N_walkers,d))
    w0 /= np.linalg.norm(w0, axis=-1, keepdims=True)  # Normalize student directions

    if mode == 'positive':
        # Ensure random but positive overlap with teacher
        dot = np.dot(w0, w_star)
        w0 -= (1 - np.sign(dot)[:, None]) * dot[:, None] * w_star[None, :]
    elif mode == 'fixed':
        # Ensure fixed overlap with teacher
        w0 -= np.dot(w0, w_star)[:, None] * w_star[None, :]
        w0 /= np.linalg.norm(w0, axis=-1, keepdims=True)
        w0 = m0 * w_star + np.sqrt(1 - m0**2) * w0

    if N_walkers == 1:
        w0 = w0[0]  # Return a single vector if only one run

    return w_star, w0



# Define the system of ODEs: dm/dt = f1(m, h, params), dh/dt = f2(m, h, params)
def velocity_field(s , state, params):
    """
    Compute the velocity field for the ODE system.
    Parameters:
    -----------
    state : list or array
        Current state [m, h].
    s : float
        Defines current in log scale as t = d^s.
    params : dict
        Dictionary of parameters needed for the computation.
    Returns:
    --------
    [dm/dt, dh/dt] : list
        Derivatives.
    """
    # print(type(params), params)
    m, h = state
    # Decomposing parameters
    d = params['d']
    # s_ref = params['s_ref']
    # alpha = params['alpha']
    p = params['p']
    k = params['k']
    tau = params['tau']
    teacher_fun = activations[params['teacher']][0]
    student_deriv = activations[params['student']][1]
    h_star = params['h_star']
    c = params['c']
    scale = params['scale']
    # if scale == 'log':
    #     # p = p0*d**(-gam0 - xp*s*(k-1))
    #     p = p0/(1  +  d**(0.25*(k-1))*(s/s_ref)**alpha)**2
    # else:
    #     p = p0*s**(-gam0- xp*(k-1))
    # Compute auxiliary quantities
    g = teacher_fun(h_star)
    v_w = p*g*student_deriv(h)*(h_star-h*m) + (1-p)*c*k*m**(k-1)*(1-m**2)
    v_x = p*g*student_deriv(h)*(d-h*h) + (1-p)*c*k*m**(k-1)*(h_star-m*h)
    # v_x = p*g*student_deriv(h)*d
    if scale == 'log':
        # Compute the derivatives
        dmdt = (np.log(d)*d**(s-1))/(2*d*m*tau)*v_w**2
        dhdt = (np.log(d)*d**(s-1)/(m*tau))*v_w*v_x - np.log(d)*d**(s-1)*h/(2*tau*m**2) * v_w**2
        return [dmdt, dhdt]
    elif scale == 'linear':
        # Compute the derivatives
        dmdt = v_w**2/(2*d*m*tau)
        dhdt = (v_w*v_x)/(m*d*tau) - (h*v_w**2)/(2*d*tau*m**2)
        return [dmdt, dhdt]
