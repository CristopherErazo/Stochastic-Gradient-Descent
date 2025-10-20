"""
This module contains various activation functions for neural networks.
"""

import numpy as np
import numba as nb



# Define activation functions and their derivatives
def He1(x): return x

def He1_deriv(x): return np.ones_like(x)

def He2(x): return (x**2 - 1) / np.sqrt(2)

def He2_deriv(x): return 2 * x / np.sqrt(2)

def He3(x): return (x**3 - 3 * x) / np.sqrt(6)

def He3_deriv(x): return  (3 * x**2 - 3) / np.sqrt(6)

def He4(x): return (x**4 - 6 * x**2 + 3) / np.sqrt(24)

def He4_deriv(x): return  (4 * x**3 - 12 * x) / np.sqrt(24)

def He5(x): return (x**5 - 10 * x**3 + 15 * x) / np.sqrt(120)

def He5_deriv(x): return  (5 * x**4 - 30 * x**2 + 15) / np.sqrt(120)

def relu(x): return np.maximum(0, x)

def relu_deriv(x): return 1.0 * (x > 0)

def tanh(x): return np.tanh(x)

def tanh_deriv(x): return 1 - np.tanh(x)**2

def softmax(x): return 1/(1+np.exp(-x))

def softmax_deriv(x): return softmax(x)*(1-softmax(x))




# # Define activation functions and their derivatives
# @nb.njit
# def He1(x): return x

# @nb.njit
# def He1_deriv(x): return np.ones_like(x)

# @nb.njit
# def He2(x): return (x**2 - 1) / np.sqrt(2)

# @nb.njit
# def He2_deriv(x): return 2 * x / np.sqrt(2)

# @nb.njit
# def He3(x): return (x**3 - 3 * x) / np.sqrt(6)

# @nb.njit
# def He3_deriv(x): return  (3 * x**2 - 3) / np.sqrt(6)

# @nb.njit
# def He4(x): return (x**4 - 6 * x**2 + 3) / np.sqrt(24)

# @nb.njit
# def He4_deriv(x): return  (4 * x**3 - 12 * x) / np.sqrt(24)

# @nb.njit
# def He5(x): return (x**5 - 10 * x**3 + 15 * x) / np.sqrt(120)

# @nb.njit
# def He5_deriv(x): return  (5 * x**4 - 30 * x**2 + 15) / np.sqrt(120)

# @nb.njit
# def relu(x): return np.maximum(0, x)

# @nb.njit
# def relu_deriv(x): return 1.0 * (x > 0)

# @nb.njit
# def tanh(x): return np.tanh(x)

# @nb.njit
# def tanh_deriv(x): return 1 / np.cosh(x)**2


# Dictionary to map function names to their implementations
activations = {
    "tanh": (tanh, tanh_deriv),
    "relu": (relu, relu_deriv),
    "He1": (He1, He1_deriv),
    "He2": (He2, He2_deriv),
    "He3": (He3, He3_deriv),
    "He4": (He4, He4_deriv),
    "He5": (He5, He5_deriv),
    "softmax": (softmax,softmax_deriv)
}
