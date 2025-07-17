# Mandelbrot Set: Visualization & Interaction

In this project I implement the popular [Mandelbrot Set](https://en.wikipedia.org/wiki/Mandelbrot_set) (interactive part at the bottom). It is a 2-dimensional fractal that emerges from a really simple rule. Let $c$ be an arbitrary complex number and $z=0$ (or, equivalently, $z=0+0i$). We can iteratively update $z$ using the rule:

$$
z\gets z^2+c
$$

If the magintude of $z$, i.e. $|z|=\sqrt{a^2+b^2}$, shoots to infinity as we iterate, then $c$ is not a member of the set. Otherwise, if it remains bounded by a certain value, then $c$ is a member of the set. Commonly, the number 2 is chosen as a threshold. That is, if at any point the magintude of $z$ exceeds 2, then $c$ is excluded from the set. This is the plot for different values of $c$ (real part on the x-axis and imaginary part on the y-axis):

![MandelbrotSetViz](../../assets/mandelbrot/image.png#mandelbrot_img)

It is the famous Mandelbrot set! The color of pixel $(x,y)$ represents the number of iterations it took for $z$ given $c=x+yi$ to diverge. The darker the color, the quicker $z$ diverges, white colors represent the numbers in the set. The image above is in resolution $5000\times5000$ and $100$ iterations are used to generate the set. The source code is on GitHub.

Some example trajectories are shown below:

| Iteration | $c=-1+0i$ | $c=-0.1-0.75i$ | $c=-1-1i$ |
|:---------:|:---------:|:--------------:|:---------:|
|   $z_1$   |  $0+0i$   |$-0.652-0.600i$ |$-1+1i$ |
|   $z_2$   |  $-1+0i$  |$-0.034+0.032i$ |$-1-3i$ |
|   $z_3$   |  $0+0i$   |$-0.099-0.752i$ |$-9+5i$ |
|   $z_4$   |  $-1+0i$  |$-0.655-0.599i$ |$55-91i$ |
|   $z_5$   |  $0+0i$   |$-0.029+0.036i$ |$-5257-10011i$ |
|   $z_6$   |  $-1+0i$  |$-0.100-0.752i$ |$-72584073+105255653i$ |
| $\ldots$  | $\ldots$  |    $\ldots$    |  $\ldots$  |

<br>It can be seen that $c=-1+0i$ oscillates between 2 values, $c=-0.1-0.75i$ exhibits a chaotic behavior but still remains bounded, whereas the magnitude of $c=-1-1i$ quickly explodes to very high values.

<!-- code -->

The set is generated as follows:

```python
def mandelbrot_set(
        range_real: Tuple[float, float] = (-2., 1.), 
        range_im: Tuple[float, float] = (1.5, -1.5), 
        dims: Tuple[float, float] = (5_000, 5_000), 
        max_iter: int = 100
    ) -> torch.Tensor:
```

Inputs:

- `range_real` defines the smallest and largest real parts of the numbers to be considered for the set $(x_{\min}, x_{\max})$. The default range is $[-2,1]$.
- `range_im` defines the smallest and largest imaginary parts of the numbers $(y_{\min}, y_{\max})$. The default range is $[1.5,-1.5]$.
- `dims` represents the number of real and imaginary values, i.e. the resolution of the resulting image $(w,h)$. The default dimensions are $5000\times 5000$.
- `max_iter` defines the maximum number of iterations $n$ as the membership of a number in the set is defined iteratively. The default value is $100$. Note that it is impossible to determine whether a number is in the set or not as we would need to perform infinitely many iterations. The more iterations we have, the more accurate our estimation will be. This can be experienced first-hand with the interaction in the end!

Outputs:

- The mandelbrot set as a matrix $M$ of evenly spaced complex numbers such that $M_{xy}$ is the number of iterations it takes for the magnitude of the corresponding complex number $c_{xy}$ to diverge. If it does not diverge, $M_{xy}=n$.

**Step 1**: Define all numbers to consider $C$ using the definition above.

```python
    C = (
        np.linspace(range_real[0], range_real[1], dims[1]) + 
        np.linspace(range_imag[0], range_imag[1], dims[0])[:, np.newaxis] * 1j
    )
```

**Step 2**: Initialize all numbers $Z$, which would keep track of the evolution of the numbers $C$.

```python
    Z = np.zeros_like(C)
```

**Step 3**: Initialize the Mandelbrot set $M$, which would keep the status of the numbers $C$. Initially, all numbers are in the set ($M=n$) and are continuously excluded as we iterate. As mentioned before, in the end $M_{xy}$ will be equal to the number of iterations it takes for $c_{xy}$ to diverge.

```python
    M = np.full_like(C, fill_value=max_iter, dtype=np.uint8)
```

**Step 4**: Initialize 2 boolean matrices:

- `not_diverged` keeps track of the numbers that still have not diverged before the start of the current iteration
- `diverged` keeps track of the numbers that diverge in the current iteration.

```python
    not_diverged = np.ones_like(C, dtype=bool)
    diverged = np.zeros_like(C, dtype=bool)
```

**Step 5**: Iteratively update the set.

```python
    for n in tqdm(range(max_iter)):
```

**Step 6**: Update the values of $Z$ for the numbers currently inside the set using the rule $z_{xy}\gets z_{xy}^2+c_{xy}$

```python
        Z[not_diverged] = Z[not_diverged] ** 2 + C[not_diverged]
```

**Step 7**: Find the values of $Z$ inside the set that just diverged, which are the numbers with magnitudes exceeding 2.

```python
        diverged[not_diverged] = Z[not_diverged].abs() > 2
```

**Step 8**: Update the values of the numbers that just diverged from the set.

```python
        M[not_diverged & diverged] = n
```

**Step 9**: Record the numbers that just diverged.

```python
        not_diverged &= ~diverged
```

That's it!

```python
    return M
```

<!-- code -->

This project utilizes vectorized operations from NumPy to efficiently update the set by processing all numbers $C$ at the same time. Moreover, numbers that already diverged are discarded, further accelerating the process.

## Interaction

Below one can interact with the Mandelbrot set. You can **zoom in infinitely by inscribing a square** or **click on the image** to see the trajectory of the corresponding number throughout the iterations!
