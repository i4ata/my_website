"""This module provides functionality for the Jacobian Inverse Technique"""

import numpy as np
from tqdm.auto import tqdm
import argparse
from typing import Optional

def parse_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-H', type=float, default=5e-3, help='The constant used to approximate the derivatives of the forward kinematics')
    parser.add_argument('-t', '--tolerance', type=float, default=1e-3, help='The maximum allowed distance between a solution and the ground truth')
    parser.add_argument('--max_steps', type=int, default=10, help='The maximum number of iterations of the JIT')
    return parser.parse_args()

def fk_2d(lengths: np.ndarray, angles: np.ndarray, save_all: bool = False) -> np.ndarray:
    """
    This function solves the forward kinematics problem in 2D.
    Works with an arbitrary number of batched dimensions

    Inputs:
    
    - lengths: NumPy array with shape (..., N), 
    representing the lengths of the N links in each system

    - angles: NumPy array with shape (..., N),
    representing the angles in radians for each of the N joints in each system

    - save_all: bool
    If True, return the location and absolute angle of each joint (..., <x,y,theta>, N)
    If False, return only the endeffector position (..., 2)
    """

    # Initialize the accumulators (all 0)
    x, y, theta = np.zeros(shape=(3, *angles.shape[:-1]))
    if save_all: xs, ys, thetas = [x.copy()], [y.copy()], [theta]
    
    # Loop over the links and compute the running positions and angles
    for i in range(lengths.shape[-1]):
        theta += angles[..., i]
        x += lengths[..., i] * np.cos(theta)
        y += lengths[..., i] * np.sin(theta)
        
        if save_all: xs.append(x.copy()), ys.append(y.copy()), thetas.append(theta.copy())

    # Return the result
    return (
        np.stack((x,y), axis=-1) 
        if not save_all else 
        np.stack([np.stack(arr, axis=-1) for arr in (xs, ys, thetas)])
    )

def fk_3d(lengths: np.ndarray, angles: np.ndarray, save_all: bool = False) -> np.ndarray:
    """
    This function solves the forward kinematics problem in 3D.
    Works with an arbitrary number of batched dimensions

    Inputs:
    
    - lengths: NumPy array with shape (..., N), 
    representing the lengths of the N links in each system

    - angles: NumPy array with shape (..., 2N),
    representing the angles in radians for each of the N joints in each system

    - save_all: bool
    If True, return the location and absolute angle of each joint (..., <x,y,z,theta,phi>, N)
    If False, return only the endeffector position (..., 3)
    """

    # Initialize the accumulators
    x, y, z, theta, phi = np.zeros(shape=(5, *angles.shape[:-1]))
    if save_all: xs, ys, zs, thetas, phis = [x.copy()], [y.copy()], [z.copy()], [theta.copy()], [phi.copy()]
    
    # Reshape the angles from (..., 2N) to (..., N, <theta,phi>)
    angles = angles.reshape(*angles.shape[:-1], -1, 2)

    # Iterate over the links and compute the running positions and angles
    for i in range(lengths.shape[-1]):
        theta += angles[..., i, 0]
        phi += angles[..., i, 1]

        x += lengths[..., i] * np.cos(theta) * np.sin(phi)
        y += lengths[..., i] * np.sin(theta) * np.sin(phi)
        z += lengths[..., i] * np.cos(phi)

        if save_all:
            xs.append(x.copy()), ys.append(y.copy()), zs.append(z.copy()), 
            thetas.append(theta.copy()), phis.append(phi.copy())

    # Return the result
    return (
        np.stack((x,y,z), axis=-1)
        if not save_all else
        np.stack([np.stack(arr, axis=-1) for arr in (xs, ys, zs, thetas, phis)])
    )

def JIT(
        lengths: np.ndarray, 
        ee_true: np.ndarray,
        max_steps: int = 100, 
        h: float = 5e-3, 
        tolerance: float = 1e-3,
        save_all: bool = False
    ) -> np.ndarray:
    """
    This function performs the Jacobian Inverse Technique for invserse kinematics.
    Works with batched and unbatched inputs

    Inputs:
    - lengths: NumPy array (N) or (B,N), 
    representing the lengths of the N links
    
    - ee_true: NumPy array (B,2) or (2) or (B,3) or (3), 
    representing the desired endeffector positions

    - max_steps (100): Maximum number of iterations for which to perform the algorithm
    - h (5e-3): The constant h used for computing the Jacobian
    - tolerance (1e-3): The threshold for considering the problem as solved
    - save_all (False): 
    If True, return the predicted angles at each iteration
    If False, return the final optimal angles for each sample
    """

    # Ensure both the lengths and endeffector positions are either batched or unbatched
    assert lengths.ndim == ee_true.ndim

    # The endeffector position is either a 2D or 3D point
    assert ee_true.shape[-1] in (2, 3)

    # Handle cases where the input is unbatched. Add a batch dimension
    if lengths.ndim == 1:
        lengths = lengths[np.newaxis]
        ee_true = ee_true[np.newaxis]

    # Infer the dimensions
    dimensions = ee_true.shape[1]
    fk = fk_2d if dimensions == 2 else fk_3d

    # Define random angles to start with (sampled uniformly from [-pi, pi])
    angles_shape = (len(ee_true), lengths.shape[1] * (dimensions - 1))
    angles = (np.random.random(size=angles_shape) * 2 - 1) * np.pi

    if save_all: angles_preds = [angles.copy()]

    # Define the matrix H defined as hI that comes in later
    H = h * np.eye(angles_shape[1])

    # Iterate
    for step in tqdm(range(max_steps)):

        # Compute the endeffector positions given the current angles
        # [B, {2,3}]
        predictions = fk(lengths, angles)

        # Compute the error vector
        # [B, {2,3}]
        error = ee_true - predictions
        
        # Find the samples have not converged yet
        # [B], bool
        not_converged_mask: np.ndarray = np.linalg.norm(error, axis=1) > tolerance
        
        if not not_converged_mask.any(): break
        
        # Filter out only those samples and work only with them from now on
        # [<=B, {N, 2N}]
        unknown_angles = angles[not_converged_mask]
        
        # Compute all combinations of angles with tiny changes to one component
        # [<=B, {N, 2N}, {N, 2N}]
        perturbed_angles = (

            # [<=B, {N, 2N}, {N, 2N}]
            np.tile(unknown_angles[:, np.newaxis], (1, unknown_angles.shape[1], 1)) 
            
            + 
            
            # [{N, 2N}, {N, 2N}]
            H[:unknown_angles.shape[1], :unknown_angles.shape[1]]
        )

        # Find the positions with each perturbation of the angles
        # [<=B, {N, 2N}, {2, 3}]
        perturbed_positions = fk(

            # [<=B, 1, N] or [<=B, 1, 2N]
            lengths=lengths[not_converged_mask][:, np.newaxis], 
            
            # [<=B, N, N] or [<=B, 2N, 2N]
            angles=perturbed_angles
        )
        
        # Compute the Jacobian using the definition of a derivative
        # [<=B, {2, 3}, {N, 2N}]
        jacobian_matrix = (

            # [<=B, N, 2] or [<=B, 2N, 3]
            perturbed_positions 

            - 
            
            # [<=B, 1, 2] or [<=B, 1, 3]
            predictions[not_converged_mask][:, np.newaxis]
        
        ).transpose(0, -1, -2) / h

        # Compute the update step for the angles
        # [<=B, {N, 2N}]
        delta_angles = (

            # [<=B, N, 2] or [<=B, 2N, 3]
            np.linalg.pinv(jacobian_matrix) 
            
            @
            
            # [<=B, 2, 1] or [<=B, 3, 1]
            error[not_converged_mask][..., np.newaxis]
        
        ).squeeze(-1)
        
        # Update the angles and normalize them between -pi and pi
        angles[not_converged_mask] += delta_angles
        angles[not_converged_mask] = normalize_angles(angles[not_converged_mask])
        if save_all: angles_preds.append(angles.copy())
    
    # Return the result
    return (
        reshape_angles(angles, dimensions)
        if not save_all else 
        np.stack(angles_preds, axis=1)
    ).squeeze(0) # Remove batched dimension if only 1 sample

def reshape_angles(angles: np.ndarray, dimensions: int) -> np.ndarray:
    return (
        angles
        if dimensions == 2 else
        angles.reshape(*angles.shape[:-1], -1, 2)
    )

def normalize_angles(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2 * np.pi) - np.pi
