"""
This module contains various utility function.
"""

import os , inspect, pickle
import numpy as np
from .functions import activations
from datetime import datetime
import subprocess


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


# Data , files and paths handling

def sanitize(v):
    "Round float values to 4 digits"
    if isinstance(v, float):
        return f"{v:.4g}"  # round to 4 significant digits
    return str(v).replace(' ', '-')

def dict_to_name(dic, sep='_', key_value_sep=''):
    """Convert a dictionary of parameters into a consistent string."""
    items = sorted(dic.items())  # sort for reproducibility
    return sep.join(f"{k}{key_value_sep}{sanitize(v)}" for k, v in items)

def make_paths_general(base_dir,subfolder_names, file_name ,dic = None ,ext=None):
    """
    Create a consistent folder structure for experiment results.
    """
    # Turn parameter dictionary into a friendly name
    if dic is not None:
        param_str = dict_to_name(dic)
        filename = f'{file_name}_{param_str}'
    else:
        param_str = ''
        filename = file_name
    
    if ext is not None: filename += '.' + ext
    # Build full paths
    if isinstance(subfolder_names,list):
        dir_path = os.path.join(base_dir, *subfolder_names)
    else:
        dir_path = os.path.join(base_dir, subfolder_names)

    file_path = os.path.join(dir_path,filename)
    # Ensure the directory exists
    os.makedirs(dir_path, exist_ok=True)
    return file_path , filename, dir_path

def make_data_paths(file_name, experiment_name= '', params=None,ext='pkl',base_dir='./data'):

    if params == None or not 'fixed' in params.keys() :
        subfolder_names = experiment_name
        params_file = params
    elif 'fixed' in params.keys() and 'variable' in params.keys():
        subfolder_names = [experiment_name,dict_to_name(params['fixed'])]
        params_file = params['variable']

    file_path, filename , dir_path = make_paths_general(base_dir,subfolder_names, file_name ,params_file ,ext=ext)

    return file_path , filename, dir_path


def make_params_dict(*names):
    "Make a dictionary with the variables in names and the values they have."
    caller_locals = inspect.currentframe().f_back.f_locals
    if len(names) == 1:
        dict = {name: caller_locals[name] for name in names[0]}
    elif len(names) == 2:
        dict = {'fixed': {name: caller_locals[name] for name in names[0]},
                'variable': {name: caller_locals[name] for name in names[1]}}
    else:
        raise ValueError(f'Revise names provided, maximum 2 inputs. Got {names = }')
    return dict

def save_fig(fig, file_name, params=None, show = True, ext='png',base_dir="../plots",date=False,bbox_inches='tight',dpi=200):
    subfolder_names = datetime.now().strftime("%Y-%m") if date else ''
    file_path, filename , dir_path = make_paths_general(base_dir,subfolder_names, file_name ,params ,ext=ext)
    fig.savefig(file_path, dpi=dpi,bbox_inches=bbox_inches)
    if show:
        print(f'Figure saved on {dir_path} as {filename}')



def save_data(data, file_name, experiment_name= '', params=None,show=True,ext='pkl',base_dir='./data'):

    file_path, filename , dir_path = make_data_paths(file_name, experiment_name, params,ext,base_dir)

    if ext == 'txt':
        if not isinstance(data,np.ndarray):
            raise TypeError(f'Data must be np array, got {type(data)}.')
        elif data.ndim > 2:
            raise TypeError('Array must be at most 2D to save as txt.')
        else:
            np.savetxt(file_path,data)
            message = 'File saved with np.savetxt'
    
    elif ext == 'npy':
        if not isinstance(data,np.ndarray):
            raise TypeError(f'Data must be np array, got {type(data)}.')    
        else:
            np.save(file_path,data)
            message = 'File saved with np.save'

    else:
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
        message = 'File saved with pickle.dump'

    if show:
        print(f'{message} on {dir_path} as {filename}')



def load_data(file_name, experiment_name= '', params=None,show=True,ext='pkl',base_dir='../data'):
    
    if params == None or not 'fixed' in params.keys() :
        subfolder_names = experiment_name
        params_file = params
    elif 'fixed' in params.keys() and 'variable' in params.keys():
        subfolder_names = [experiment_name,dict_to_name(params['fixed'])]
        params_file = params['variable']

    file_path, filename , dir_path = make_paths_general(base_dir,subfolder_names, file_name ,params_file ,ext=ext)

    if ext == 'txt':
        data = np.loadtxt(filename)
        message = 'loaded with np.loadtxt'
    
    elif ext == 'npy':
        data = np.load(filename)
        message = 'loaded with np.load'

    else:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        message = 'loaded with pickle.load'

    if show:
        print(f'File {filename} {message} from {dir_path}')
    
    return data



def download_cluster_data(server_name,path_cluster,path_local,filename_cluster,filename_local=None,show=True):

    # Define available servers
    servers = {'ulysses':'cerazova@frontend2.hpc.sissa.it:~/',
                'peralba':'cerazova@peralba.phys.sissa.it:/u/c/cerazova/'}
    if server_name not in ['ulysses','peralba']:
        raise ValueError('Invalid server name')
    else:
        server = servers[server_name]
    
    # If filename_local not provided use the same as in cluster
    if filename_local is None : 
        filename_local = filename_cluster

    # Construct paths
    cluster = os.path.join(server, path_cluster ,filename_cluster).replace('\\','/')
    local = os.path.join(path_local,filename_local)
    # Check if file exist locally
    if os.path.exists(local):
        print(f'File already exist: {local}')
        return
    else: # create local directory if not existing
        os.makedirs(path_local, exist_ok=True)
    

    # Run scp command to copy from cluster
    result = subprocess.run(['scp', cluster, local], capture_output=True, text=True)

    # Report
    if result.stderr:
        # print(f'File not found: {cluster}')
        print('Error:')
        print(result.stderr)
    else:
        if show: 
            print(f'SUCCESS LOADING FILE: {filename_cluster}')