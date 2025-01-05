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

    x, y, theta = np.zeros(shape=(3, *angles.shape[:-1]))
    if save_all: xs, ys, thetas = [x.copy()], [y.copy()], [theta]
    
    for i in range(lengths.shape[-1]):
        theta += angles[..., i]
        x += lengths[..., i] * np.cos(theta)
        y += lengths[..., i] * np.sin(theta)
        
        if save_all: xs.append(x.copy()), ys.append(y.copy()), thetas.append(theta.copy())

    return (
        np.stack((x,y), axis=-1) 
        if not save_all else 
        np.stack([np.stack(arr, axis=-1) for arr in (xs, ys, thetas)])
    )

def fk_3d(lengths: np.ndarray, angles: np.ndarray, save_all: bool = False) -> np.ndarray:
    
    x, y, z, theta, phi = np.zeros(shape=(5, *angles.shape[:-1]))
    angles = angles.reshape(*angles.shape[:-1], -1, 2)
    if save_all: xs, ys, zs, thetas, phis = [x.copy()], [y.copy()], [z.copy()], [theta.copy()], [phi.copy()]

    for i in range(lengths.shape[-1]):
        theta += angles[..., i, 0]
        phi += angles[..., i, 1]

        x += lengths[..., i] * np.cos(theta) * np.sin(phi)
        y += lengths[..., i] * np.sin(theta) * np.sin(phi)
        z += lengths[..., i] * np.cos(phi)

        if save_all:
            xs.append(x.copy()), ys.append(y.copy()), zs.append(z.copy()), 
            thetas.append(theta.copy()), phis.append(phi.copy())

    return (
        np.stack((x,y,z), axis=-1)
        if not save_all else
        np.stack([np.stack(arr, axis=-1) for arr in (xs, ys, zs, thetas, phis)])
    )

def JIT(
        lengths: np.ndarray, 
        ee_true: np.ndarray,
        angles: Optional[np.ndarray] = None,
        max_steps: int = 100, 
        h: float = 5e-3, 
        tolerance: float = 1e-3,
        save_all: bool = False
    ) -> np.ndarray:

    assert lengths.ndim == ee_true.ndim
    assert ee_true.shape[-1] in (2, 3)
    if lengths.ndim == 1:
        lengths = lengths[np.newaxis]
        ee_true = ee_true[np.newaxis]

    dimensions = ee_true.shape[1]
    angles_shape = (len(ee_true), lengths.shape[1] * (dimensions - 1))
    if angles is None:
        angles = (np.random.random(size=angles_shape) * 2 - 1) * np.pi
    assert angles.shape == angles_shape

    if save_all: angles_preds = [angles.copy()]

    fk = fk_2d if dimensions == 2 else fk_3d
    H = h * np.eye(angles.shape[1])

    for step in tqdm(range(max_steps)):

        # [n, dimensions]
        predictions = fk(lengths, angles)

        # [n, dimensions]
        error = ee_true - predictions
        
        # [n], bool
        not_converged_mask: np.ndarray = np.linalg.norm(error, axis=1) > tolerance
        
        if not not_converged_mask.any(): break
        
        # [<=n, num_angles]
        unknown_angles = angles[not_converged_mask]
        
        # [<=n, num_angles, num_angles]
        perturbed_angles = (

            # [<=n, num_angles, num_angles]
            np.tile(unknown_angles[:, np.newaxis], (1, unknown_angles.shape[1], 1)) + 
            
            # [num_angles, num_angles]
            H[:unknown_angles.shape[1], :unknown_angles.shape[1]]
        )

        # [<=n, num_angles, dimensions]
        perturbed_positions = fk(lengths=lengths[not_converged_mask][:, np.newaxis], angles=perturbed_angles)
        
        # [<=n, dimensions, num_angles]
        jacobian_matrix = (perturbed_positions - predictions[not_converged_mask][:, np.newaxis]).transpose(0, -1, -2) / h
        
        # [<=n, num_angles]
        delta_angles = (np.linalg.pinv(jacobian_matrix) @ error[not_converged_mask][..., np.newaxis]).squeeze(-1)
        
        angles[not_converged_mask] += delta_angles
        angles[not_converged_mask] = normalize_angles(angles[not_converged_mask])
        if save_all: angles_preds.append(angles.copy())
    
    return (
        reshape_angles(angles, dimensions)
        if not save_all else 
        np.stack(angles_preds, axis=1)
    ).squeeze(0)

def reshape_angles(angles: np.ndarray, dimensions: int) -> np.ndarray:
    return (
        angles
        if dimensions == 2 else
        angles.reshape(*angles.shape[:-1], -1, 2)
    )

def normalize_angles(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2 * np.pi) - np.pi

if __name__ == '__main__':
    
    np.random.seed(0)
    args = parse_args()

    # lengths = np.ones((10, 4))
    # ee_true = np.ones((10, 2))
    # angles = JIT(lengths=lengths, ee_true=ee_true, max_steps=args.max_steps, h=args.H, tolerance=args.tolerance)
    
    # plot_2d(lengths=lengths, ee_true=ee_true, angles=angles[-1])

    lengths = np.ones(4)
    ee_true = np.array([1,1,1])
    angles = JIT(lengths=lengths, ee_true=ee_true, max_steps=args.max_steps, h=args.H, tolerance=args.tolerance, save_all=True)
    print(angles.shape)
    # plot_3d(lengths=lengths, ee_true=ee_true, angles=angles[-1])
