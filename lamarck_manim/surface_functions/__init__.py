"""
Surface Functions Module

This module provides mathematical surface functions for 3D visualization.
All functions follow the signature: func(u, v) -> np.array([x, y, z])

Functions:
    rastrigin_func: Rastrigin function - highly multimodal optimization test function
"""

import numpy as np


def rastrigin_func(u, v, A=10, n=2, scale=0.03):
    """
    Rastrigin function: f(x,y) = A*n + Σ(xi² - A*cos(2π*xi))
    Highly multimodal function with many local minima.
    
    Global minimum at (0, 0) with value 0.
    Has many regularly distributed local minima, making it challenging for gradient descent.
    
    Args:
        u, v: Input coordinates (x, y)
        A: Amplitude parameter (default 10)
        n: Number of dimensions (default 2)
        scale: Scaling factor for z values (reduced to 0.03 for better visualization)
    
    Returns:
        numpy array: [x, y, z] where z is the function value scaled
    """
    x = u
    y = v
    z = (A * n + (x**2 - A * np.cos(2 * np.pi * x)) + (y**2 - A * np.cos(2 * np.pi * y))) * scale
    return np.array([x, y, z])


__all__ = ['rastrigin_func']
