"""This script visualizes the Mandelbrot Set"""

import numpy as np
from tqdm.auto import tqdm
import argparse
from typing import Tuple

def parse_args() -> argparse.Namespace:
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--range_real', nargs=2, type=float, default=(-2., 1.), help='Range of real part of c')
    parser.add_argument('--range_imag', nargs=2, type=float, default=(-1.5, 1.5), help='Range of imaginary part of c')
    parser.add_argument('--dims', nargs=2, type=int, default=(5_000, 5_000), help='Dimensions of the resulting image (level of precision)')
    parser.add_argument('--max_iter', type=int, default=100, help='Number of iterations')
    parser.add_argument('--file', type=str, default=None, help='Filename to store the resulting image')

    return parser.parse_args()

def mandelbrot_set(
        range_real: Tuple[float, float] = (-2., 1.), 
        range_imag: Tuple[float, float] = (-1.5, 1.5), 
        dims: Tuple[float, float] = (5_000, 5_000), 
        max_iter: int = 100,
        save_all: bool = False,
    ) -> np.ndarray:
    """
    This function computes the values in the Mandelbrot Set

    Inputs:
     <min_real, max_real>: The range of the real part
     <min_imaginary, max_imaginary>: The range of the imaginary part
     <w,h>: The precision of the resulting set. The higher the better
     max_iter: The number of iterations performed
    """

    # The grid of all imaginary numbers to consider [h x w]
    # C[x, y] = (x_min + x*delta) + (y_max - y*delta)i
    C = (
        np.linspace(range_real[0], range_real[1], dims[1]) + 
        np.linspace(range_imag[1], range_imag[0], dims[0])[:, np.newaxis] * 1j
    )
    
    # The resulting image to fill up
    M = np.full_like(C, fill_value=max_iter, dtype=np.uint8)
    
    # Keep track of the value of z for each imaginary number
    Z = np.zeros_like(C)

    # Keep track of numbers that still haven't diverged
    not_diverged = np.ones_like(C, dtype=bool)

    # Keep track of numbers that just diverged
    diverged = np.zeros_like(C, dtype=bool)
    
    # Optionally, keep track of all intermediate steps
    if save_all: frames = [M]

    for n in tqdm(range(max_iter)):

        # Update z using the rule z = z^2 + c
        Z[not_diverged] = Z[not_diverged] ** 2 + C[not_diverged]
        
        # If |z| > 2, the magnitude of the number will diverge to infinity
        diverged[not_diverged] = np.abs(Z[not_diverged]) > 2
        
        # Update the values of the numbers that just diverged
        # Set them to the number of iterations it took
        M[not_diverged & diverged] = n

        # Record the numbers that just diverged
        not_diverged &= ~diverged
        
        if save_all: frames.append(M.copy())

    return M if not save_all else np.stack(frames)

if __name__ == '__main__':
    
    import matplotlib.pyplot as plt

    args = parse_args()
    m = mandelbrot_set(
        range_real=args.range_real, 
        range_imag=args.range_imag,
        dims=args.dims,
        max_iter=args.max_iter,
        save_all=False
    )
    
    plt.imshow(m, extent=(*args.range_real, *args.range_imag), cmap='hot')
    plt.title('Mandelbrot Set')
    plt.xlabel('Re(c)')
    plt.ylabel('Im(c)')
    plt.tight_layout()
    if args.file is not None: plt.savefig(args.file, dpi=1_000)
    plt.show()
