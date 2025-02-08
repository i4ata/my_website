import numpy as np
from typing import Literal, Optional, Tuple
from tqdm.auto import tqdm
import argparse
np.seterr(divide='ignore', invalid='ignore')

def parse_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', type=int, default=3, help='The number of bodies in the simulation')
    parser.add_argument('-d', type=int, default=2, choices=(2, 3), help='The number of dimensions in which the simulation runs')
    parser.add_argument('--step', type=float, default=.005, help='The difference between 2 timesteps in the simulation')
    parser.add_argument('--max_t', type=int, default=1_000, help='The number of steps in the simulation')
    parser.add_argument('-G', type=float, default=1, help='Gravitational constant for the system')
    parser.add_argument('--max_magnitude', type=float, default=15, help='The bound for the magnitude of the acceleration')
    return parser.parse_args()

def _compute_acceleration(
    x: np.ndarray, 
    M: np.ndarray, 
    G: float = 1, 
    max_magnitude: float = 15
) -> np.ndarray:
    """
    This function computes each body's acceleration relative to the other bodies using Newton's law of motion
    
    Inputs
    - x: Numpy array of shape [b, n, d], representing the bodies' current positions
    - M: Numpy array of shape [b, n, 1] or [n, 1], representing the bodies' masses
    - G: Newton's gravitational constant
    - max_magnitude: Number to clamp the accelerations

    Returns
    - Numpy array of shape [b, n, d], representing the bodies' current acceleration in each direction
    """

    # [b, n, n, d]
    # [..., i, j] represents the vector from body j to body to i   
    directions = x[:, np.newaxis] - x[:, :, np.newaxis]

    # [b, n, n, 1]
    # [..., i, j] represents m_i * m_j
    masses = (M[:, np.newaxis] * M[:, np.newaxis])[:, :, :, np.newaxis]

    # [b, n, n, 1]
    # [..., i, j] represents the distances between bodies i and j
    distances: np.ndarray = np.linalg.norm(directions, axis=3, keepdims=True)

    # [b, n, d]
    # [..., i, j] represents the acceleration of body i in direction j
    # Computed using Newton's law of motion
    accelerations = np.nansum(directions * masses * G / distances ** 3, axis=2)
    
    # Clamp the acceleration's magnitudes
    accelerations = _clamp_magnitude(accelerations, max_magnitude=max_magnitude)

    return accelerations

def _clamp_magnitude(a: np.ndarray, max_magnitude: float = 15) -> np.ndarray:
    """
    This function clamps the accelerations' magnitudes if they are larger than a hyperparameter

    Inputs
    - a: Numpy array of shape [b, n, d], representing the bodies' current accelerations
    - max_magnitude: Largest allowed acceleration
    """

    # [b, n]
    # Compute the acceleration magnitudes for each body
    magnitudes: np.ndarray = np.linalg.norm(a, axis=-1)
    
    # Wherever the acceleration is larger than the maximum magnitude, clamp it
    # by first normalizing it and then multiplying by the maximum magnitude
    large = magnitudes > max_magnitude
    a[large] *= max_magnitude / magnitudes[large][..., np.newaxis]
    
    return a

def run_euler(
    b: int = 1, 
    n: int = 3, 
    d: Literal[2,3] = 2,
    init_position: Optional[np.ndarray] = None,
    init_velocity: Optional[np.ndarray] = None,
    M: Optional[np.ndarray] = None,
    step: float = .005, 
    max_t: int = 1_000,
    G: float = 1,
    max_magnitude: float = 15
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    This function generates the N body simulation

    Inputs
    - b: Number of simulations to run in parallel
    - n: Number of bodies in each simulation
    - d: Number of dimensions in the siulation (Can technically be any number)
    - init_position: Numpy array of shape [b, n, d] or [n, d] that defines the initial positions of the bodies. If unspecified, each value is sampled from U[-1,1]
    - init_velocity: Numpy array of shape [b, n, d] or [n, d] that defines the initial positions of the bodies. If unspecified, each value is sampled from U[-1,1]
    - M: Numpy array of shape [b, n] or [n] that defines the masses of the bodies. If unspecified, all masses are 1
    - step: Delta time, the step size in the simulation
    - max_t: Total number of timesteps for the simulation
    - G: Newton's gravitational constant
    - max_magnitude: The maximum allowed acceleration in the system. Used to prevent close encounters

    Returns:
    - Numpy array of shape [b, t, n, d], representing the bodies positions over time
    - Numpy array of shape [b, t, n, d], representing the bodies velocities over time
    - Numpy array of shape [b, t, n, d], representing the bodies accelerations over time
    """

    # [b, t, n, d] -> [simulation id, timestep, body id, dimension id]
    x, v, a = np.zeros((3, b, max_t, n, d))
    x[:, 0] = np.random.rand(n, d) * 2 - 1 if init_position is None else init_position
    v[:, 0] = np.random.rand(n, d) * 2 - 1 if init_velocity is None else init_velocity

    # Body masses
    M = np.ones((b, n)) if M is None else np.atleast_2d(M)
    
    # Loop over time
    for t in tqdm(range(max_t - 1)):

        # Euler's integration step
        a[:, t] = _compute_acceleration(x[:, t], M=M, G=G, max_magnitude=max_magnitude)
        x[:, t+1] = x[:, t] + v[:, t] * step
        v[:, t+1] = v[:, t] + a[:, t] * step
        
    return x.squeeze(0), v.squeeze(0), a.squeeze(0)

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    args = parse_args()
    x, v, a = run_euler(
        n=args.n, 
        d=args.d, 
        step=args.step, 
        max_t=args.max_t, 
        G=args.G, 
        max_magnitude=args.max_magnitude
    )
    for i in range(args.n):
        x_, y_ = x[:, i, 0], x[:, i, 1]
        plt.plot(x_, y_)
        plt.scatter(x_[0], y_[0], s=20)
    plt.show()
