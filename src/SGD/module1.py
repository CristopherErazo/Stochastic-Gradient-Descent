"""
This module contains various utility functions and definitions for neural network training,
including activation functions, their derivatives, and gradient computation functions.

Author  : Cristopher Erazo <cerazova@sissa.it>
Date    : August 2025
Version : 1.5
"""

import numpy as np
import time



# Activation functions and their derivatives
He1 = lambda x: x
He1_deriv = lambda x: np.ones_like(x)
He1_deriv2 = lambda x: np.zeros_like(x)
He1_deriv3 = lambda x: np.zeros_like(x)
He1_deriv4 = lambda x: np.zeros_like(x)

He2 = lambda x: (x**2 - 1)/np.sqrt(2)
He2_deriv = lambda x:  2*x/np.sqrt(2)
He2_deriv2 = lambda x: 2*np.ones_like(x)/np.sqrt(2)
He2_deriv3 = lambda x: np.zeros_like(x)
He2_deriv4 = lambda x: np.zeros_like(x)

He3 = lambda x: (x**3 - 3 * x)/np.sqrt(6)
He3_deriv = lambda x: (3 * x**2 - 3)/ np.sqrt(6)
He3_deriv2 = lambda x: (6*x)/ np.sqrt(6)
He3_deriv3 = lambda x: (6) * np.ones_like(x) / np.sqrt(6)
He3_deriv4 = lambda x: np.zeros_like(x)

He4 = lambda x: (x**4 - 6 * x**2 + 3)/np.sqrt(24)
He4_deriv = lambda x: (4 * x**3 - 12 * x)/np.sqrt(24)
He4_deriv2 = lambda x: (12*x**2 - 12)/np.sqrt(24)
He4_deriv3 = lambda x: (24*x)/np.sqrt(24)
He4_deriv4 = lambda x: (24)*np.ones_like(x)/np.sqrt(24) 

He5 = lambda x: (x**5 - 10*x**3 + 15*x)/np.sqrt(120)
He5_deriv = lambda x: (5*x**4 - 30*x**2 + 15)/np.sqrt(120)
He5_deriv2 = lambda x: (20*x**3 - 60*x )/np.sqrt(120)
He5_deriv3 = lambda x: (60*x**2 - 60)/np.sqrt(120)
He5_deriv4 = lambda x: 120*x/np.sqrt(120)

relu = lambda x: np.maximum(0, x)
relu_deriv = lambda x: 1.0 * (x > 0)
relu_deriv2 = lambda x: 1e4 * (np.abs(x) < 0.5*1e-4)#np.zeros_like(x)
relu_deriv3 = lambda x: np.zeros_like(x)
relu_deriv4 = lambda x: np.zeros_like(x)

tanh = lambda x: np.tanh(x)#/np.sqrt(0.394294)
tanh_deriv = lambda x: (1 / np.cosh(x)**2)#/np.sqrt(0.394294)
tanh_deriv2 = lambda x: -2* (1/np.cosh(x))**2 * np.tanh(x)
tanh_deriv3 = lambda x: 4*np.tanh(x)**2 *(1/np.cosh(x))**2 - (1/np.cosh(x))**4
tanh_deriv4 = lambda x: 16*np.tanh(x) *(1/np.cosh(x))**4 - 8*np.tanh(x)**3 * (1/np.cosh(x))**2     

fun1 = lambda x: (He2(x) + He4(x))/np.sqrt(2)
fun1_deriv = lambda x: (He2_deriv(x) + He4_deriv(x))/np.sqrt(2)
fun1_deriv2 = lambda x: (He2_deriv2(x) + He4_deriv2(x))/np.sqrt(2)

fun2 = lambda x: (He1(x) + He2(x) + He4(x))/np.sqrt(3)
fun2_deriv = lambda x: (He1_deriv(x) + He2_deriv(x) + He4_deriv(x))/np.sqrt(3)
fun2_deriv2 = lambda x: (He1_deriv2(x) + He2_deriv2(x) + He4_deriv2(x))/np.sqrt(3)

fun3 = lambda x: (He1(x) + He2(x) - He4(x))/np.sqrt(3)
fun3_deriv = lambda x: (He1_deriv(x) + He2_deriv(x) - He4_deriv(x))/np.sqrt(3)
fun3_deriv2 = lambda x: (He1_deriv2(x) + He2_deriv2(x) - He4_deriv2(x))/np.sqrt(3)

fun4 = lambda x: (He1(x) - He2(x) + He4(x))/np.sqrt(3)
fun4_deriv = lambda x: (He1_deriv(x) - He2_deriv(x) + He4_deriv(x))/np.sqrt(3)
fun4_deriv2 = lambda x: (He1_deriv2(x) - He2_deriv2(x) + He4_deriv2(x))/np.sqrt(3)


def IE(function):
    if function == 'tanh' or function =='He1':
        k = 1
    elif function == 'He2':
        k = 2
    elif function == 'He3':
        k = 3
    elif function == 'He4':
        k = 4
    return k 

def alphas_lr_scales(d,k):
    if k <=0: raise ValueError('k must be >=1')
    if k==1:
        ac = 1.0
    elif k==2:
        ac = np.log(d)
    else:
        ac = 1.0*d**(k-2)
    
    alpha = ac*(np.log(d))**(min(3,k)-1)
    lr = alpha**(-3/4)/d
    return alpha , lr, ac




functions = {"tanh": [tanh,tanh_deriv,tanh_deriv2,tanh_deriv3,tanh_deriv4],
             "relu": [relu,relu_deriv,relu_deriv2,relu_deriv3,relu_deriv4],
             "He1": [He1,He1_deriv,He1_deriv2,He1_deriv3,He1_deriv4],
             "He2": [He2,He2_deriv,He2_deriv2,He2_deriv3,He2_deriv4],
             "He3": [He3,He3_deriv,He3_deriv2,He3_deriv3,He3_deriv4], 
             "He4": [He4,He4_deriv,He4_deriv2,He4_deriv3,He4_deriv4],
             "He5": [He5,He5_deriv,He5_deriv2,He5_deriv3,He5_deriv4],
             "fun1": [fun1,fun1_deriv,fun1_deriv],
             "fun2": [fun2,fun2_deriv,fun2_deriv2],
             "fun3": [fun3,fun3_deriv,fun3_deriv2],
             "fun4": [fun4,fun4_deriv,fun4_deriv2]}


def eval_teacher(w_star, X, fun):
    '''
    Evaluate the teacher function on the input data X using the teacher weights w_star.
    
    w_star : (d,) array, teacher weights |w_star| = 1
    X : (N_runs,n,d) or (n,d) or (n,N_runs,d) array, input data
    
    Returns:
    Y : (N_runs,n) or (n,) or (n,N_runs) array, output of the teacher function
    '''
    preac = X @ w_star   # (N_runs,n) or (n,) or (n,N_runs)
    Y = fun(preac)  # (N_runs,n) or (n,) or (n,N_runs)
    return Y


def init_w(d,N_runs = 1, m0 = 0. ,mode='random'):
    '''
    Initialize the teacher and student directions. 
    Both directions are sampled in the d-dimensional sphere 
    of radius 1, with the student direction having an overlap 
    of  (w* . w ) =  m0 with the teacher direction.

    Input:
    d : int, dimension of the space
    N_runs : int, number of runs (independent students)
    m0 : float, desired overlap between teacher and student directions
    mode: str, either random, fixed or positive

    Returns:
    w_star : (d,) array, teacher direction
    w0 : (N_runs, d) array, student directions
    -----------
    '''
    # Sample teacher direction
    w_star = np.random.normal(size=d)
    w_star /= np.linalg.norm(w_star)
    
    # Sample student directions at initialization
    w0 = np.random.normal(size=(N_runs,d))
    if mode == 'random': # purely random initialization
        w0 /= np.linalg.norm(w0,axis=-1)[:,None]
    elif mode == 'positive': #ensure random but positive overlap with teacher
        w0 /= np.linalg.norm(w0,axis=-1)[:,None]
        dot = w0 @ w_star
        w0 = w0 - (1-np.sign(dot)[:,None])*dot[:,None]*w_star[None,:]
        # print(signs)
    elif mode == 'fixed':  # with overlap = m0 with teacher
        w0 = w0 - (w0 @ w_star)[:,None] * w_star[None,:]
        w0 /= np.linalg.norm(w0,axis=-1)[:,None]
        w0 = m0 * w_star + np.sqrt(1-m0**2)*w0
    
    if N_runs == 1:
        w0 = w0[0]
    return w_star , w0

# BATCH GRADIENT FUNCTIONS

def batch_grad_mse(w,X,Y,fun,deriv,spherical=False):
    '''
    Compute the gradient of the MSE loss function for a batch of data.
    
    w : (N_runs,d) or (d,) array, weights for each run
    X : (N_runs,n,d)  or (n,d) array, input data for each run
    Y : (N_runs,n) or (n,) array, target values for each run
    fun : callable, function to compute predictions
    deriv : callable, derivative of the function fun
    spherical : bool, if true project out the radial component of gradient
    
    Returns:
    grad : (N_runs,d) or (d,) array, gradients for each run
    '''
    if len(X.shape) == 3: # Then X.shape = (N_runs,n,d) and w.shape = (N_runs,d)
        # Compute preactivations, predictions, errors, and gradients
        preac = np.sum(w[:,None,:] * X,axis=-1)   # (N_runs,n)
        pred = fun(preac)  # (N_runs,n)
        err = Y - pred  # (N_runs,n)
        grad = - (err * deriv(preac))[:,:,None] * X   # (N_runs,n,d)
        grad = np.sum(grad,axis=1)  # (N_runs,d) -- w.shape = (N_runs,d)
        if spherical:
            dot = np.sum(w * grad, axis=1) # (N_runs)
            grad -= dot[:,None] * w
        return grad
    
    else: # Then X.shape = (n,d) and  w.shape =(d)
        # Compute preactivations, predictions, errors, and gradients
        preac = np.sum(w[None,:] * X,axis=-1)  # (n)
        pred = fun(preac)  # (n)
        err = Y - pred  # (n)
        grad = - (err * deriv(preac))[:,None] * X # (n,d)
        grad = np.sum(grad,axis=0) # (d) --  w.shape =(d)
        if spherical: grad -= np.dot(w,grad) * w
        return grad


def batch_grad_corr(w,X,Y,fun,deriv,spherical=False):
    '''
    Compute the gradient of the correlation loss function for a batch of data.
    
    w : (N_runs,d) or (d,) array, weights for each run
    X : (N_runs,n,d)  or (n,d) array, input data for each run
    Y : (N_runs,n) or (n,) array, target values for each run
    fun : callable, function to compute predictions
    deriv : callable, derivative of the function fun
    spherical : bool, if true project out the radial component of gradient
        
    Returns:
    grad : (N_runs,d) or (d,) array, gradients for each run
    '''
    if len(X.shape) == 3: # Then X.shape = (N_runs,n,d) and w.shape = (N_runs,d)
        # Compute preactivations, predictions, errors, and gradients
        preac = np.sum(w[:,None,:] * X,axis=-1)  # (N_runs,n)
        grad = - (Y * deriv(preac))[:,:,None] * X # (N_runs,n,d)
        grad = np.sum(grad,axis=1)  # (N_runs,d) -- w.shape = (N_runs,d)
        if spherical:
            dot = np.sum(w * grad, axis=1) # (N_runs)
            grad -= dot[:,None] * w
        return grad
    
    else: # Then X.shape = (n,d) and w.shape = (d)
        # Compute preactivations, predictions, errors, and gradients
        preac = np.sum(w[None,:] * X,axis=-1)  # (n)
        grad = - (Y * deriv(preac))[:,None] * X # (n,d)
        grad = np.sum(grad,axis=0) # (d) --  w.shape =(d)
        if spherical: grad -= np.dot(w,grad) * w
        return grad
    
def gradient_descent(d, n, epochs , lr, teacher , student, loss = 'mse', online = True, normalize = False, spherical = False,N_runs=1, m0=0.,compute_loss=False):
    '''
    Perform gradient descent to train a student model to mimic a teacher model.
    d : int, dimension of the input data
    n : int, number of samples in the training set
    epochs : int, number of training epochs
    lr : float, learning rate for gradient descent
    teacher : str, name of the teacher function among ('tanh', 'relu', 'He2', 'He3', 'He4')
    student : str, name of the student function among ('tanh', 'relu', 'He2', 'He3', 'He4')
    loss : str, loss function to use ('mse' or 'corr')
    online : bool, whether to use online learning (True) or multipass learning (False)
    normalize : bool, whether to normalize student weights to a sphere of radius norm after each update
    spherical : bool, if true project out the radial component of gradient.    
    N_runs : int, number of independent runs (students)
    m0 : float, desired overlap between teacher and student directions at initialization  (w* . w )/|w*||w| =  m0
    compute_loss: float , whether to compute the test loss or not - (extra computation time)
   
    Returns:
    overlap : (epochs+1,) or (epochs+1, N_runs) array, cosine-similarity of the student with respect to the teacher at each epoch 
    norm_w : (epochs+1,) or (epochs+1, N_runs) array, norm of the student weights at each epoch (if normalized will be = np.ones(overlap.shape) )
    test_error : (epochs+1,) or (epochs+1, N_runs) array, test error of the student on the teacher function at each epoch / n
    w_student : (N_runs,d) or (d,) array, final student weights after training
    -----------
    '''
    # Check for consistency
    if spherical and not normalize:
        raise ValueError("spherical gradient only available when normalized dynamics!")
        
    # Define the teacher function and its derivative
    fun_teacher = functions[teacher][0]
    fun, deriv = functions[student][:2]
    # Select loss
    if loss == 'mse':
        gradient = batch_grad_mse
    elif loss == 'corr':
        gradient = batch_grad_corr
    else:
        raise ValueError("Loss function not recognized. Use 'mse' or 'corr'.")
    # Define shapes 
    if N_runs == 1:
        shpe = (epochs+1,)
        shpex = (n,d)
    else:
        shpe = (epochs+1, N_runs)
        shpex = (N_runs, n, d)
    
    # Initialize weigths and data (train and test)
    w_teacher, w_student = init_w(d, N_runs, m0=m0)
    X_train = np.random.normal(size=shpex) 
    Y_train = eval_teacher(w_teacher, X_train , fun_teacher)

    X_test = np.random.normal(size=shpex)  # (N_runs,n,d)
    Y_test = eval_teacher(w_teacher, X_test , fun_teacher)


    # Initialize arrays to store results
    overlap = np.zeros(shape=shpe) 
    norm_w = np.zeros(shape=shpe)  
    if compute_loss:
        test_error = np.zeros(shape=shpe)
    else:
        test_error = None
    
    # Compute parameters
    t = 0

    norm_w[t] = np.linalg.norm(w_student, axis=-1) # (N_runs) or scalar
    overlap[t] = w_student @ w_teacher / norm_w[t]  # (N_runs) or scalar
    
    if compute_loss:
        # Compute the loss on the training set
        if N_runs > 1:
            preac = np.sum(w_student[:,None,:] * X_test,axis=-1) 
        else:
            preac = np.sum(w_student[None,:] * X_test,axis=-1) 
        Y_pred = fun(preac)  # (N_runs,n) or (n,)
        if loss == 'mse':
            test_error[t] = 0.5*np.mean((Y_test - Y_pred)**2, axis=-1)  # (N_runs) or scalar
        elif loss == 'corr':
            test_error[t] = 1 - np.mean(Y_test * Y_pred, axis=-1)  # (N_runs) or scalar

    # Loop over epochs
    for t in range(epochs):
        if online and t > 0: # Sample new data for online learning
            X_train = np.random.normal(size=shpex) 
            Y_train = eval_teacher(w_teacher, X_train , fun_teacher)
        # Compute the gradient 
        grad = gradient(w_student, X_train, Y_train, fun, deriv, spherical=spherical)
        w_student -= lr * grad  # Update student weights

        # Normalize the student weights if normalized with norm |w|=1
        if normalize:
            n_w = np.linalg.norm(w_student, axis=-1)
            if N_runs > 1: n_w = n_w[:, None]  # Ensure shape compatibility
            w_student /= n_w

        
        # Compute parameters
        norm_w[t+1] = np.linalg.norm(w_student, axis=-1) # (N_runs) or scalar
        overlap[t+1] = w_student @ w_teacher / norm_w[t+1] # (N_runs) or scalar


        # Compute the loss on the training set
        if compute_loss:
            if N_runs > 1:
                preac = np.sum(w_student[:,None,:] * X_test,axis=-1)
            else:
                preac = np.sum(w_student[None,:] * X_test,axis=-1) 
            Y_pred = fun(preac)  # (N_runs,n) or (n,)
            if loss == 'mse':
                test_error[t+1] = 0.5*np.mean((Y_test - Y_pred)**2, axis=-1)  # (N_runs) or scalar
            elif loss == 'corr':
                test_error[t+1] = 1 - np.mean(Y_test * Y_pred, axis=-1)  # (N_runs) or scalar
        
    return overlap , norm_w , test_error, w_student

def test_loss_aligned(d,n,teacher,student,w_teacher,loss,sign):
   
    X_test = np.random.normal(size=(n,d))  # (N_runs,n,d)
    fun_teacher = functions[teacher][0]
    fun_student = functions[student][0]
    Y_test = eval_teacher(w_teacher, X_test , fun_teacher)
    Y_pred = eval_teacher(sign*w_teacher,X_test,fun_student)
    if loss == 'mse':
        test_error= 0.5*np.mean((Y_test - Y_pred)**2, axis=-1)  # (N_runs) or scalar
    elif loss == 'corr':
        test_error= 1 - np.mean(Y_test * Y_pred, axis=-1)  # (N_runs) or scalar
    return test_error

#######################################################################
#######################################################################
# ONLINE GRADIENT FUNCTIONS
#######################################################################
#######################################################################

def online_grad_mse_runs(w,x,y,fun,deriv,spherical=False):
    '''
    Compute the gradient of the MSE loss function for a single point
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
    pred = fun(preac)  # (N_runs)
    err = y - pred  # (N_runs)
    grad = - (err * deriv(preac))[:,None] * x   # (N_runs,d)
    if spherical:
        dot = np.sum(w * grad, axis=1) # (N_runs)
        grad -= dot[:,None] * w
    return grad  # (N_runs,d)

def online_grad_mse(w,x,y,fun,deriv,spherical=False):
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
    preac = w @ x  # scalar
    pred = fun(preac)  # scalar
    err = y - pred  # scalar
    grad = - (err * deriv(preac))* x   # (d,)
    if spherical: grad -= np.dot(w,grad) * w
    return grad

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


def online_gradient_descent(d, n, epochs , lr, teacher , student, p_reset = 0.0 , loss = 'mse', online = True, normalize = False,spherical=False, N_runs=1, m0=0.,pts_epoch = 10,shuffle=True,compute_loss=False):
    '''
    Perform gradient descent to train a student model to mimic a teacher model.
    d : int, dimension of the input data
    n : int, number of samples in the training set
    epochs : int, number of training epochs
    lr : float, learning rate for gradient descent
    teacher : str, name of the teacher function among ('tanh', 'relu', 'He2', 'He3', 'He4')
    student : str, name of the student function among ('tanh', 'relu', 'He2', 'He3', 'He4')
    p_reset : float (0,1), probability of reset to initial condition.
    loss : str, loss function to use ('mse' or 'corr')
    online : bool, whether to use online learning (True) or multipass learning (False)
    normalize : bool, whether to normalize student weights to a sphere of radius norm after each update
    spherical: bool, whether to project out the radial component of grad or not
    N_runs : int, number of independent runs (students)
    m0 : float, desired overlap between teacher and student directions at initialization  (w* . w )/|w*||w| =  m0
    pts_epoch : int, number of points to save in results at each epoch. Should be a divisor of n.
    shuffle : bool, whether to shuffle the data at each epoch (only for online learning)
    compute_loss: float , whether to compute the test loss or not - (extra computation time)
   
    Returns:
    **tot_pts = pts_epoch * epochs # Total number of points to save**

    overlap : (tot_pts,) or (tot_pts, N_runs) array, overlap of the student with respect to the teacher at each epoch / norm**2
    norm_w : (tot_pts,) or (tot_pts, N_runs) array, norm of the student weights at each epoch / norm
    test_error : (tot_pts,) or (tot_pts, N_runs) array, test error of the student on the teacher function at each epoch / n
    w_student : (N_runs,d) or (d,) array, final student weights after training
    -----------
    '''
    # Check for consistency
    if spherical:
        if not normalize:
            raise ValueError("spherical gradient only available when normalized dynamics!")
        
    # Define the teacher function and its derivative
    fun_teacher = functions[teacher][0]
    fun, deriv = functions[student][:2]
    # Select loss
    if loss == 'mse':
        if N_runs > 1:
            gradient = online_grad_mse_runs # (N_runs,d)
        else:
            gradient = online_grad_mse # (d,)
    elif loss == 'corr':
        if N_runs > 1:
            gradient = online_grad_corr_runs # (N_runs,d)
        else:
            gradient = online_grad_corr # (d,)
    else:
        raise ValueError("Loss function not recognized. Use 'mse' or 'corr'.")
    # Define shapes 
    
    if pts_epoch > n or pts_epoch <= 0 or n % pts_epoch != 0:
        raise ValueError("pts_epoch must be less than or equal to n, greater than 0, and a divisor of n.")
    tot_pts = pts_epoch * epochs + 1  # Total number of points to save
    print_every = int(n / pts_epoch)  # Number of points to save at each epoch
    if N_runs == 1:
        shpe = (tot_pts,)
        shpex = (n,d)
    else:
        shpe = (tot_pts, N_runs)
        shpex = (n, N_runs, d)
    
    # Initialize weigths and data (train and test)
    w_teacher, w_student_init = init_w(d, N_runs, m0=m0) # (d,) and (N_runs,d) or (d,)
    X_train = np.random.normal(size=shpex) # (n,d) or (n, N_runs, d)
    Y_train = eval_teacher(w_teacher, X_train , fun_teacher) # (n,) or (n,N_runs) 

    X_test = np.random.normal(size=shpex)  # (n,d) or (n,N_runs,d)
    Y_test = eval_teacher(w_teacher, X_test , fun_teacher) # (n,) or (n,N_runs)

    w_student = w_student_init.copy()

    # Initialize arrays to store results
    overlap = np.zeros(shape=shpe) 
    norm_w = np.zeros(shape=shpe) 
    if compute_loss: 
        test_error = np.zeros(shape=shpe)
    else:
        test_error = None

    # Compute parameters
    t = 0
    norm_w[t] = np.linalg.norm(w_student, axis=-1) # (N_runs) or scalar
    overlap[t] = w_student @ w_teacher / norm_w[t]  # (N_runs) or scalar

    if  compute_loss :
        # Compute the loss on the training set
        if N_runs > 1:         
            preac = np.sum(w_student[None,:,:] * X_test,axis=-1) # (n,N_runs)
        else:                   
            preac = np.sum(w_student[None,:] * X_test,axis=-1)  # (n,)
        Y_pred = fun(preac)  # (n) or (n,N_runs)
        if loss == 'mse':
            test_error[t] = 0.5*np.mean((Y_test - Y_pred)**2, axis=0)  # (N_runs) or scalar
        elif loss == 'corr':
            test_error[t] = 1 - np.mean(Y_test * Y_pred, axis=0)  # (N_runs) or scalar

    # Loop over epochs
    for epc in range(epochs):
        if online and t > 0: # Sample new data for online learning
            X_train = np.random.normal(size=shpex) 
            Y_train = eval_teacher(w_teacher, X_train , fun_teacher)
        if shuffle:
            idx = np.random.permutation(n)  # Shuffle indices for online learning
        else:
            idx = np.arange(n)
        # Loop over points in the epoch
        for i in range(n):
            ix = idx[i]
            x_train = X_train[ix]  # (N_runs,d) or (d,)
            y_train = Y_train[ix]  # (N_runs) or scalar


            #Check if reset
            if np.random.random() < p_reset:
                w_student = w_student_init.copy()
                print(epc,i)
            else:
                # Compute the gradient 
                grad = gradient(w_student, x_train, y_train, fun, deriv,spherical=spherical) # (N_runs,d) or (d,)
                w_student -= lr * grad  # Update student weights

            # Normalize the student weights if normalize
            if normalize: 
                n_w = np.linalg.norm(w_student, axis=-1) # (N_runs,) or scalar
                if N_runs > 1: n_w = n_w[:, None]  # Ensure shape compatibility
                w_student /= n_w         

            if (i+1) % print_every == 0:
                t += 1  # Increment epoch counter 
                # Compute parameters
                
                norm_w[t] = np.linalg.norm(w_student, axis=-1) # (N_runs) or scalar
                overlap[t] = (w_student @ w_teacher) / norm_w[t] # (N_runs) or scalar

                if compute_loss :
                    # Compute the loss on the training set
                    if N_runs > 1:         
                        preac = np.sum(w_student[None,:,:] * X_test,axis=-1)  # (n,N_runs)
                    else:                   
                        preac = np.sum(w_student[None,:] * X_test,axis=-1)  # (n,)
                    Y_pred = fun(preac)  # (n) or (n,N_runs)
                    if loss == 'mse':
                        test_error[t] = 0.5*np.mean((Y_test - Y_pred)**2, axis=0)  # (N_runs) or scalar
                    elif loss == 'corr':
                        test_error[t] = 1 - np.mean(Y_test * Y_pred, axis=0)  # (N_runs) or scalar

    return overlap , norm_w , test_error, w_student


###############################################################
# New simplified version of spherical online sgd with no epochs ala Ben Arous
###############################################################


def online_gradient_descent_arous(w_teacher, w0_student, d, n, lr, teacher , student, t_prints, epsilon = 0.0, loss = 'mse',
                            normalize = True,spherical=True, compute_loss=False , n_test = None, p_reset = 0.0,i_print=None):
    '''
    Perform gradient descent to train a student model to mimic a teacher model.
    w_teacher: (d,) array, teacher direction
    w0_student: (N_runs,d) or (d,) array, student vector
    d : int, dimension of the input data
    n : int, number of total samples
    lr : float, learning rate for gradient descent
    teacher : str, name of the teacher function among ('tanh', 'relu', 'He2', 'He3', 'He4')
    student : str, name of the student function among ('tanh', 'relu', 'He2', 'He3', 'He4')
    t_prints : array, values of the time steps where we save the data
    epsilon : float, standard deviation of the noise level in the data
    loss : str, loss function to use ('mse' or 'corr')
    normalize : bool, whether to normalize student weights to a sphere of radius norm after each update
    spherical: bool, whether to project out the radial component of grad or not
    compute_loss: float , whether to compute the test loss or not - (extra computation time)
    n_test: int, if compute_loss=True then n_test is the number of test points
    p_reset : float (0,1), probability of reset to initial condition.
   
    Returns:
    **tot_pts = pts_epoch * epochs # Total number of points to save**

    overlap : (tot_pts,) or (tot_pts, N_runs) array, overlap of the student with respect to the teacher at each epoch / norm**2
    norm_w : (tot_pts,) or (tot_pts, N_runs) array, norm of the student weights at each epoch / norm
    test_error : (tot_pts,) or (tot_pts, N_runs) array, test error of the student on the teacher function at each epoch / n
    w_student : (N_runs,d) or (d,) array, final student weights after training
    -----------
    '''
    if len(w0_student.shape) == 2: # multiple parallel runs
        N_runs , d = w0_student.shape
    else:
        d = len(w0_student)
        N_runs = 1

    # Check for consistency
    if spherical:
        if not normalize:
            raise ValueError("spherical gradient only available when normalized dynamics!")
        
    # Define the teacher function and its derivative
    fun_teacher = functions[teacher][0]
    fun, deriv = functions[student][:2]

    # Select loss
    if loss == 'mse':
        if N_runs > 1:
            gradient = online_grad_mse_runs # (N_runs,d)
        else:
            gradient = online_grad_mse # (d,)
    elif loss == 'corr':
        if N_runs > 1:
            gradient = online_grad_corr_runs # (N_runs,d)
        else:
            gradient = online_grad_corr # (d,)
    else:
        raise ValueError("Loss function not recognized. Use 'mse' or 'corr'.")
    
    
    tot_pts = len(t_prints)
    if (np.diff(t_prints) <= 0).any():
        raise ValueError('t_prints must be strictly increasing function')
    if t_prints[0]<0 or t_prints[-1]>n:
        raise ValueError('t_prints must be between [0,n]')
    if t_prints[-1] != n:
        raise ValueError('last t_print should be = n, otherwise wasted computation')
    t_prints = t_prints.astype(int) # ensure integer type
    
    # Define shapes 
    if N_runs == 1:
        shpe_save = (tot_pts,)
        shpex = (d)
    else:
        shpe_save = (tot_pts, N_runs)
        shpex = (N_runs,d)
    

    w_student = w0_student.copy()

    # Initialize arrays to store results
    overlap = np.zeros(shape=shpe_save) 
    norm_w = np.zeros(shape=shpe_save) 

    if compute_loss: 
        X_test = np.random.normal(size=(d,n_test)) 
        Y_test = eval_teacher(w_teacher, X_test.T , fun_teacher).T # (n_test,) or (N_runs,n_test)
        test_error = np.zeros(shape=shpe_save)
    else:
        test_error = None


    n_print = 0
    t_print = t_prints[n_print]

    

    for t in range(n+1): 
        # Check wheather to print result or not
        if t == t_print:
            if not i_print == None:
                if n_print%i_print == 0: print(f'{n_print}/{tot_pts}   ,  {100*t/n :.5} %')
            # Compute parameters
            norm_w[n_print] = np.linalg.norm(w_student, axis=-1) # (N_runs) or scalar
            overlap[n_print] = (w_student @ w_teacher) / norm_w[n_print] # (N_runs) or scalar

            if compute_loss :
                # Compute the loss on the training set | remember: Y_test.shape = (n_test,) or (N_runs,n_test) and X_test.shape = (d,n_test)
                Y_pred = fun(w_student @ X_test) #(n_test) or (N_runs , n_test)
                if loss == 'mse':
                    test_error[n_print] = 0.5*np.mean((Y_test - Y_pred)**2, axis=-1)  # (N_runs) or scalar
                elif loss == 'corr':
                    test_error[n_print] = 1 - np.mean(Y_test * Y_pred, axis=-1)  # (N_runs) or scalar
            # Increment epoch counter 
            if t < n:
                n_print += 1  
                t_print = t_prints[n_print]

        # ONLINE GRADIENT DESCENT
        x_train = np.random.normal(size=shpex) 
        y_train = eval_teacher(w_teacher,x_train,fun_teacher) + epsilon*np.random.randn(N_runs)

        #Check if reset
        if np.random.random() < p_reset:
            w_student = w0_student.copy()
            print(t)
        else:
            # Compute the gradient 
            grad = gradient(w_student, x_train, y_train, fun, deriv,spherical=spherical) # (N_runs,d) or (d,)
            w_student -= lr * grad  # Update student weights

        # Normalize the student weights if normalize
        if normalize: 
            n_w = np.linalg.norm(w_student, axis=-1) # (N_runs,) or scalar
            if N_runs > 1: n_w = n_w[:, None]  # Ensure shape compatibility
            w_student /= n_w   

    return overlap , norm_w , test_error, w_student


############################################################
############################################################
#                REPEATING SAMPLES
############################################################
############################################################



def coefficients(a,b,teacher,student):

    g = functions[teacher][0]
    sigs = functions[student]

    ga = g(a)
    sbs = [sigs[k](b) for k in range(len(sigs))]
    # print(sbs)

    c0 = 0.5*(ga-sbs[0])**2
    c1 = -a*sbs[1]*(ga - sbs[0])
    c2 = (ga-sbs[0])*(b*sbs[1]-a*a*sbs[2]) + a*a*sbs[1]**2
    c3 = - (ga-sbs[0])*(a**3*sbs[3]-3*a*b*sbs[2]) + 3*a*sbs[1]*(a*a*sbs[2]-b*sbs[1])
    return np.array([c0,c1,c2/2,c3/6])

def coefficients_function(a,b,function):
    fun = functions[function]
    sum = (a+b)
    dif = (a-b)
    m = sum/np.sqrt(2)
    dfs = [fun[k](m) for k in range(len(fun))]

    c0 = dfs[0]
    c1 = dif * dfs[1]/(2*np.sqrt(2))  
    c2 = ( dif**2 * dfs[2] - np.sqrt(2) * sum * dfs[1] )/16
    c3 = ( np.sqrt(2) * dif**2 * dfs[3]  - 6 * sum * dfs[2] + 6*np.sqrt(2)*dfs[1] ) * dif/192
    c4 = ( 6*(5*a**2 - 6*a*b + 5*b**2)*dfs[2] + dif**4 * dfs[4] - 6*np.sqrt(2)*sum*dif**2 * dfs[3] - 30*np.sqrt(2)*sum*dfs[1] )/1536
    return np.array([c0,c1,c2,c3,c4])

def coeff_product(coeff_p , coeff_q):
    kmax = min(len(coeff_p),len(coeff_q))
    coeff_pq = np.array( [np.sum(coeff_p[:k+1] * coeff_q[:k+1][::-1],axis=0) for k in range(kmax)] )
    return coeff_pq

def coefficients_symmetric(a,b,teacher,student):

    coeff_teacher = coefficients_function(a,b,teacher)
    coeff_student = coefficients_function(a,-b,student)

    coeff_err = coeff_teacher - coeff_student 
    coeff_loss = 0.5*coeff_product(coeff_err,coeff_err)

    # teach_teach = coeff_product(coeff_teacher,coeff_teacher)
    # stud_stud = coeff_product(coeff_student,coeff_student)
    # teach_stud = coeff_product(coeff_teacher,coeff_student)

    # coeff_loss = 0.5*(teach_teach - 2*teach_stud + stud_stud)
    return coeff_loss

def coefficients_hermite_apx(a,b,teacher,student): 
    hermites = ['He1','He2','He3','He4','He5']
    g = functions[teacher][0](a)
    sig = functions[student][0](b)
    coeff = []
    for her in hermites:
        h = functions[her][0]
        ha = h(a)
        hb = h(b)
        coeff.append( np.mean(g*ha,axis=0)*np.mean(sig*hb,axis=0))
    return -np.array(coeff)



def preactivation_derivatives(a,b,type='original',teach=True):
    '''
    type = {'original','symmetric','simplified'}
    '''
    zero = np.zeros_like(a)
    if type == 'symmetric':
        if teach:
            return np.array([a,b/2,-a/4,3*b/8,-15*a/16])
        else:
            return np.array([b,a/2,-b/4,3*a/8,-15*b/16])
    if type == 'simplified':
        if teach:
            return np.array([(a+b),(a-b)/2,-(a+b)/4,3*(a-b)/8,-15*(a+b)/16])/np.sqrt(2)
        else:
            return np.array([(a-b),(a+b)/2,-(a-b)/4,3*(a+b)/8,-15*(a-b)/16])/np.sqrt(2)
    if type == 'original': 
        if teach:
            return np.array([a,zero,zero,zero,zero])
        else:
            return np.array([b,a,-b,zero,-3*b])


def function_derivatives(a,b,function,type='original',teach=True):
    h = preactivation_derivatives(a,b,type=type,teach=teach)
    fun = functions[function]
    df = [fun[k](h[0]) for k in range(len(fun))]

    f_0 = df[0]    
    f_1 = h[1]*df[1]
    f_2 = h[1]**2 * df[2] + h[2]*df[1]
    f_3 = df[3]* h[1]**3 + 3*h[1]*h[2]*df[2] + h[3]*df[1]
    f_4 = df[4]* h[1]**4 + 6*df[3]*h[1]**2*h[2] + df[2]*(3* h[2]**2 + 4*h[3]*h[1]) + h[4]*df[1]     
    return np.array([f_0,f_1,f_2,f_3,f_4])

def difference_derivatives(a,b,teacher,student,type='original'): 
    teach = function_derivatives(a,b,teacher,type=type,teach=True)
    stud = function_derivatives(a,b,student,type=type,teach=False)
    return teach-stud

def loss_derivatives(a,b,teacher,student,type='original'): 
    D = difference_derivatives(a,b,teacher,student,type=type)
    
    L_0 = D[0]**2/2
    L_1 = D[0]*D[1]
    L_2 = D[1]**2 + D[0]*D[2]
    L_3 = 3*D[1]*D[2] + D[0]*D[3]
    L_4 = 3*D[2]**2 + 4*D[1]*D[3] + D[0]*D[4]
    return np.array([L_0,L_1,L_2/2,L_3/6,L_4/24])


def create_dataset(N,d,epsilon,w_teacher,teacher):
    if N == 0:
        return None, None
    else:
        fun_teacher = functions[teacher][0]
        x_train = np.random.normal(size=(N,d)) 
        y_train = eval_teacher(w_teacher,x_train,fun_teacher) + epsilon*np.random.randn(N)
        dataset = [x_train,y_train]
        return dataset 

def eval_phi_repeat(w,dataset,student,loss):
    X_train , Y_train = dataset
    fun = functions[student][0]
    preac = np.einsum('id,djk->ijk', X_train, w)
    if loss == 'mse':
        loss_values = 0.5*(Y_train[:,None,None] - fun(preac))**2
    elif loss == 'corr':
        loss_values = 1 - Y_train[:,None,None] * fun(preac)
    return loss_values.mean(axis=0)

def eval_phi_repeat_projection(m,Y_train,a,b,student,loss):
    fun = functions[student][0]
    preac = a[:,None]*m[None,:] + b[:,None]*np.sqrt(1-m**2)[None,:]
    if loss == 'mse':
        loss_values = 0.5*(Y_train[:,None] - fun(preac))**2
    elif loss == 'corr':
        loss_values = 1 - Y_train[:,None] * fun(preac)
    return loss_values.mean(axis=0)    
    

# def pop_loss_repetitions(projections,teacher,student,loss):
#     a , b = projections
#     N = len(a)
#     teacher_fun = functions[teacher][0]
#     student_fun = functions[student][0]
#     def phi_rep(m):
#         if loss == 'mse':
#             phi = 0.5*(teacher_fun(a[None,:])-student_fun(a[None,:]*m[:,None]+ np.sqrt(1-m[:,None]**2)*b[None:,]))**2
#         elif loss == 'corr':
#             phi = 1 - teacher_fun(a[None,:])*student_fun(a[None,:]*m[:,None]+ np.sqrt(1-m[:,None]**2)*b[None:,])
#         return np.mean(phi,axis=-1)
#     return phi_rep


def online_gradient_descent_repetition(w_teacher, w0_student, d, n, lr, teacher , student, t_prints, epsilon = 0.0, loss = 'mse', dataset = None,
                            p_repeat = 0.0 ,normalize = True,spherical=True, compute_loss=False , n_test = None,i_print=None):
    '''
    Perform gradient descent to train a student model to mimic a teacher model.
    * Inputs:
    - w_teacher: (d,) array, teacher direction
    - w0_student: (d,) array, student vector
    - d : int, dimension of the input data
    - n : int, number of total samples
    - lr : float, learning rate for gradient descent
    - teacher : str, name of the teacher function among ('tanh', 'relu', 'He2', 'He3', 'He4')
    - student : str, name of the student function among ('tanh', 'relu', 'He2', 'He3', 'He4')
    - t_prints : array, values of the time steps where we save the data
    - epsilon : float, standard deviation of the noise level in the data
    - loss : str, loss function to use ('mse' or 'corr')
    - dataset : None or list of tuples [(x[i],y[i]) for i = 1,...,N] array of the dataset to be used for repetitions.
    - p_repeat : float, if data not None - probability of usind a datapoint from dataset.
    - normalize : bool, whether to normalize student weights to a sphere of radius norm after each update
    - spherical: bool, whether to project out the radial component of grad or not
    - compute_loss: float , whether to compute the test loss or not - (extra computation time)
    - n_test: int, if compute_loss=True then n_test is the number of test points
   
    * Returns:
    **tot_pts = pts_epoch * epochs # Total number of points to save**

    - overlap : (tot_pts,) array, overlap of the student with respect to the teacher at each epoch / norm**2
    - norm_w : (tot_pts,) array, norm of the student weights at each epoch / norm
    - test_error : (tot_pts,) array, test error of the student on the teacher function at each epoch / n
    - w_student :  (d,) array, final student weights after training
    -----------
    '''

    if len(w0_student.shape) > 1:
        raise ValueError('Only one run available N_runs=1')
    d = len(w0_student)
    N_runs = 1

    # Check for consistency
    if spherical:
        if not normalize:
            raise ValueError("spherical gradient only available when normalized dynamics!")
        
    # Define the teacher function and its derivative
    fun_teacher = functions[teacher][0]
    fun, deriv = functions[student][:2]

    # Select loss
    if loss == 'mse':
        gradient = online_grad_mse # (d,)
    elif loss == 'corr':
        gradient = online_grad_corr # (d,)
    else:
        raise ValueError("Loss function not recognized. Use 'mse' or 'corr'.")
    
    
    tot_pts = len(t_prints)
    if (np.diff(t_prints) <= 0).any():
        raise ValueError('t_prints must be strictly increasing function')
    if t_prints[0]<0 or t_prints[-1]>n:
        raise ValueError('t_prints must be between [0,n]')
    if t_prints[-1] != n:
        raise ValueError('last t_print should be = n, otherwise wasted computation')
    t_prints = t_prints.astype(int) # ensure integer type
    
    # Define shapes 
    shpe_save = (tot_pts,)
    shpex = (d)
    X_train , Y_train = dataset
    N = len(Y_train)


    w_student = w0_student.copy()

    # Initialize arrays to store results
    overlap = np.zeros(shape=shpe_save) 
    norm_w = np.zeros(shape=shpe_save) 

    if compute_loss: 
        X_test = np.random.normal(size=(d,n_test)) 
        Y_test = eval_teacher(w_teacher, X_test.T , fun_teacher).T # (n_test,) or (N_runs,n_test)
        test_error = np.zeros(shape=shpe_save)
    else:
        test_error = None


    n_print = 0
    t_print = t_prints[n_print]

    t_repeats = []

    for t in range(n+1): 
        # Check wheather to print result or not
        if t == t_print:
            if not i_print == None:
                if n_print%i_print == 0 or t == n: print(f'{n_print}/{tot_pts}   ,  {100*t/n :.5} %')
            # Compute parameters
            norm_w[n_print] = np.linalg.norm(w_student) #  scalar
            overlap[n_print] = (w_student @ w_teacher) / norm_w[n_print] #  scalar

            if compute_loss :
                # Compute the loss on the training set | remember: Y_test.shape = (n_test,) and X_test.shape = (d,n_test)
                Y_pred = fun(w_student @ X_test) #(n_test) or 
                if loss == 'mse':
                    test_error[n_print] = 0.5*np.mean((Y_test - Y_pred)**2)  # scalar
                elif loss == 'corr':
                    test_error[n_print] = 1 - np.mean(Y_test * Y_pred)  # scalar
            # Increment epoch counter 
            if t < n:
                n_print += 1  
                t_print = t_prints[n_print]

        # ONLINE GRADIENT DESCENT

        #Check whether to take a new sample or repeat from dataset
        if np.random.random() < p_repeat:
            t_repeats.append(t)
            idx = np.random.choice(N)
            x_train , y_train = X_train[idx] , Y_train[idx]
        else:
            x_train = np.random.normal(size=shpex) 
            y_train = eval_teacher(w_teacher,x_train,fun_teacher) + epsilon*np.random.randn()
    
        # Compute the gradient 
        grad = gradient(w_student, x_train, y_train, fun, deriv,spherical=spherical) # (d,)
        w_student -= lr * grad  # Update student weights

        # Normalize the student weights if normalize
        if normalize: 
            n_w = np.linalg.norm(w_student) # or scalar
            w_student /= n_w   

    return overlap , norm_w , test_error, w_student , np.array(t_repeats)
