"""
Canonical problem-space surface functions.

All functions follow the signature:
    func(u, v, ...) -> np.array([x, y, z])
"""

import numpy as np


def rastrigin_func(u, v, A=10, n=2, scale=0.1):
    """
    Rastrigin benchmark surface in 2D.

    Mathematical form:
        f(x, y) = A*n + (x^2 - A*cos(2*pi*x)) + (y^2 - A*cos(2*pi*y))

    Properties:
        - Highly multimodal (many regularly spaced local minima).
        - Global minimum at (0, 0) with unscaled value 0.
        - Commonly used to test global optimization behavior.

    Args:
        u (float): X-coordinate input.
        v (float): Y-coordinate input.
        A (float, optional): Amplitude parameter controlling oscillation depth.
            Default is 10.
        n (int, optional): Nominal dimensionality constant used in the classic
            formula. In this 2D implementation, default is 2.
        scale (float, optional): Output scaling factor applied to z for
            visualization/readability. Default is 0.1.

    Returns:
        numpy.ndarray: A length-3 array ``[x, y, z]`` where ``z`` is the scaled
        Rastrigin value.
    """
    x = u
    y = v
    z = (A * n + (x**2 - A * np.cos(2 * np.pi * x)) + (y**2 - A * np.cos(2 * np.pi * y))) * scale
    return np.array([x, y, z])


def rosenbrock_func(u, v, a=1, b=100, scale=0.01):
    """
    Rosenbrock (banana) benchmark surface in 2D.

    Mathematical form:
        f(x, y) = (a - x)^2 + b * (y - x^2)^2

    Properties:
        - Narrow, curved valley that is easy to find but hard to optimize along.
        - Global minimum at (a, a^2) with unscaled value 0.
        - Standard test for optimization stability and convergence.

    Args:
        u (float): X-coordinate input.
        v (float): Y-coordinate input.
        a (float, optional): Location parameter for the global minimum in x.
            Default is 1.
        b (float, optional): Curvature/conditioning parameter. Larger values
            create a steeper valley wall. Default is 100.
        scale (float, optional): Output scaling factor applied to z for
            visualization/readability. Default is 0.01.

    Returns:
        numpy.ndarray: A length-3 array ``[x, y, z]`` where ``z`` is the scaled
        Rosenbrock value.
    """
    x = u
    y = v
    z = ((a - x) ** 2 + b * (y - x**2) ** 2) * scale
    return np.array([x, y, z])


def ackley_func(u, v, a=20, b=0.2, c=2 * np.pi, scale=0.1):
    """
    Ackley benchmark surface in 2D.

    Mathematical form:
        f(x, y) =
            -a * exp(-b * sqrt((x^2 + y^2) / 2))
            -exp((cos(c*x) + cos(c*y)) / 2)
            + a + e

    Properties:
        - Broad outer region with many local undulations.
        - Global minimum at (0, 0) with unscaled value 0.
        - Useful for testing exploration vs exploitation tradeoffs.

    Args:
        u (float): X-coordinate input.
        v (float): Y-coordinate input.
        a (float, optional): Overall basin depth/offset parameter. Default is 20.
        b (float, optional): Exponential decay parameter. Default is 0.2.
        c (float, optional): Frequency parameter for cosine terms. Default is 2*pi.
        scale (float, optional): Output scaling factor applied to z for
            visualization/readability. Default is 0.1.

    Returns:
        numpy.ndarray: A length-3 array ``[x, y, z]`` where ``z`` is the scaled
        Ackley value.
    """
    x = u
    y = v
    term1 = -a * np.exp(-b * np.sqrt((x**2 + y**2) / 2))
    term2 = -np.exp((np.cos(c * x) + np.cos(c * y)) / 2)
    z = (term1 + term2 + a + np.e) * scale
    return np.array([x, y, z])


def himmelblau_func(u, v, scale=0.01):
    """
    Himmelblau benchmark surface in 2D.

    Mathematical form:
        f(x, y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2

    Properties:
        - Multi-basin landscape with several local minima.
        - Four equal global minima with unscaled value 0 at approximately:
            (3.0, 2.0), (-2.805, 3.131), (-3.779, -3.283), (3.584, -1.848)
        - Useful when evaluating optimizer behavior on multiple equivalent optima.

    Args:
        u (float): X-coordinate input.
        v (float): Y-coordinate input.
        scale (float, optional): Output scaling factor applied to z for
            visualization/readability. Default is 0.01.

    Returns:
        numpy.ndarray: A length-3 array ``[x, y, z]`` where ``z`` is the scaled
        Himmelblau value.
    """
    x = u
    y = v
    z = ((x**2 + y - 11) ** 2 + (x + y**2 - 7) ** 2) * scale
    return np.array([x, y, z])


__all__ = [
    "rastrigin_func",
    "rosenbrock_func",
    "ackley_func",
    "himmelblau_func",
]
