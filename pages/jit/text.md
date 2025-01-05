# Jacobian Inverse Technique & Interaction

In this project I implement the [Jacobian Inverse Technique](https://en.wikipedia.org/wiki/Inverse_kinematics#The_Jacobian_inverse_technique) algorithm that is used for inverse kinematics. One can interact with it by scrolling all the way down! In the field of robotics, kinematics is the problem of finding how the links of a robot are moving in space. Naturally, this is an extremely fundamental problem and is integral of any moving robotic system. It is also relevant in animation and simulations.

When doing forward kinematics, we find the location of the end effector of a chain of links given the angles of each joint connecting two links. The analytical solution is straightforward with simple trigonometry.

## Forward Kinematics in 2D

In 2 dimensions, the $i$-th link in the chain is represented by its length $L_i$ and the angle $\theta_i$ between the link and the $x$-axis as follows:

![](../../assets/jit/2d_example.svg#jit_img)

From here, we can easily get the equation for the link's endpoint $(x_i,y_i)$:

$$
\begin{align*}
& x_i=L_i\cos(\theta_i)\\
& y_i=L_i\sin(\theta_i)
\end{align*}
$$

It is the essentially the conversion from cartesian to polar coordinates. To get the endpoint $(x^*,y^*)$ of a chain of $N$ links $\langle L_i,\theta_i \rangle, i=1,\ldots,N$ , we can simply accumulate the angles and endpositions for each consecutive link as follows:

$$
\begin{align*}
&\theta_0^\prime=x_0=y_0=0\\
&\theta_i^\prime=\theta_{i-1}^\prime+\theta_i\\
&x_i=x_{i-1}+L_i\cos(\theta_i^\prime)\\
&y_i=y_{i-1}+L_i\sin(\theta_i^\prime)\\
&x^*=x_N\\
&y^*=y_N
\end{align*}
$$

Using NumPy, the same can be done as follows:

```python
def fk_2d(lengths: np.ndarray, angles: np.ndarray) -> np.ndarray:

    # Initialize the accumulators
    x, y, theta = np.zeros(shape=(3, *angles.shape[:-1]))
    if save_all: xs, ys, thetas = [x.copy()], [y.copy()], [theta]
    
    # Accumulate for each link in the system
    for i in range(lengths.shape[-1]):

        theta += angles[..., i]
        x += lengths[..., i] * np.cos(theta)
        y += lengths[..., i] * np.sin(theta)
        
    return np.stack((x,y), axis=-1)
```

Here, the input to `fk_2d` is an array of shape $(\ldots,N)$ representing the lengths $L_i$ of the $N$ links, and another array of shape $(\ldots,N)$ representing the angles $\theta_i$ for joint. The function above calculates forward kinematics in a batched fashion with an arbitrary number of batched dimensions (why that is useful comes later), with the only constraint that the number of links $N$ is the same across all samples. The output is an array of shape $(\ldots,2)$ representing the resulting endpoints $(x^*,y^*)$ for each sample in the batch.

## Forward Kinematics in 3D

In 3 dimensions, each link is represented by 2 angles:

- $\theta$: the angle between the projection of the link on the $xy$-plane and the positive $x$-axis
- $\phi$: the angle between the link and the positive $z$-axis

![](../../assets/jit/3d_example.svg#jit_img)

From here, we can again easily get the equation for the link's endpoint $(x_i,y_i,z_i)$ using simple trigonometry:

$$
\begin{align*}
& x_i=L_i\cos(\theta_i)\sin(\phi_i)\\
& y_i=L_i\sin(\theta_i)\sin(\phi_i)\\
& z_i=L_i\cos(\phi_i)
\end{align*}
$$

Similarly to the 2D case, to get the endpoint of a chain of links, we can simply accumulate the parameters for each consecutive link, which can be done using NumPy as follows:

```python
def fk_3d(lengths: np.ndarray, angles: np.ndarray) -> np.ndarray:
    
    # Initialize the accumulators
    x, y, z, theta, phi = np.zeros(shape=(5, *angles.shape[:-1]))
    
    # Reshape the angles to [..., N, 2]
    angles = angles.reshape(*angles.shape[:-1], -1, 2)
    
    # Accumulate for each link in the system
    for i in range(lengths.shape[-1]):

        theta += angles[..., i, 0]
        phi += angles[..., i, 1]

        x += lengths[..., i] * np.cos(theta) * np.sin(phi)
        y += lengths[..., i] * np.sin(theta) * np.sin(phi)
        z += lengths[..., i] * np.cos(phi)

    return np.stack((x,y,z), axis=-1)
```

Here, the input is, again, a tensor of shape $(\ldots,N)$ representing the lengths $L_i$ of the $N$ links. However, since each link is now represented by 2 angles, the second tensor is of shape $(\ldots,2N)$ and is reshaped internally to $(\ldots,N,2)$, i.e. $(\ldots, N, \langle\theta,\phi\rangle)$. The function again allows for batched inputs with an arbitrary number of batched dimensions. The output is a tensor of shape $(\ldots,3)$ represting the resulting endpoints $(x^*,y^*,z^*)$.

**Note**: This is a considerable simplification of the problem since rotations of the joints are not considered. The Jacobian Inverse Technique as implemented here would still work as it is right now if rotations are included. This is a nice idea for a future project!

## Inverse Kinematics using the Jacobian Inverse Technique

The inverse kinematics problem is to find the angles of the joints such that the endpoint of the chain is at a desired location, i.e. how should I move my arm in such that the tip of my index finger reaches the carrot on the table. The analytical solution to this problem is way more involved compared to forward kinematics and its complexity increases exponentially with the number of links $N$. Since the problem quickly becomes intractable, it is often solved using numerical approximations. A simple yet powerful and flexible method is the Jacobian Inverse Technique. The algorithm and the NumPy implementation are as follows:

```python
def JIT(
        lengths: np.ndarray, 
        ee_true: np.ndarray,
        angles: Optional[np.ndarray] = None,
        max_steps: int = 100, 
        h: float = 5e-3, 
        tolerance: float = 1e-3
    ) -> np.ndarray:
```

The inputs are:

1. `lengths`: An array of shape $(B,N)$ representing the lengths $L_i$ of the $N$ links from each of the $B$ samples in the batch.
2. `ee_true`: An array $\xi$ of shape $(B,2)$ in the 2D case (i.e. $(B,\langle x,y \rangle)$) or $(B,3)$ in the 3D case (i.e. $(B,\langle x,y,z \rangle)$), representing the desired endeffector positions for each sample in the batch.
3. `max_steps`: Since the method is iterative, an integer defines the maximum number of iterations.
4. `h`: The constant $h$ which comes later.
5. `tolerance`: The maximum allowed distance between our solution and $\xi$.

### Step 1

```python
    dimensions = ee_true.shape[1]
    angles_shape = (len(ee_true), lengths.shape[1] * (dimensions - 1)) # (B, N) or (B, 2N)
    angles = (np.random.random(size=angles_shape) * 2 - 1) * np.pi
    fk = fk_2d if dimensions == 2 else fk_3d
```

Start off with a random prediction for the angles $\hat{\omega}_0$ of shape $(B,N)$ for the 2D case and $(B,2N)$ for the 3D case. In my case, each angle is uniformly sampled from the range $[-\pi,\pi]$. Let $p:\mathbb{R}^{\{N,2N\}}\to\mathbb{R}^{\{2,3\}}$ be the forward kinematics function for the corresponding number of dimensions (either `fk_2d` or `fk_3d`). The goal is to find the angles $\omega^*$ which minimize the error $||p(\omega^*)-\xi||$. Usually there are multiple solutions to the backward kinematics problem, i.e. there are multiple joint configurations $\omega^*$ which result in the desired endeffector position $\xi$. The Jacobian Inverse Technique finds the solution that is closest to the initial random angles $\hat{\omega}_0$.

Iteratively execute the next steps:

```python
    for step in range(max_steps):
```

### Step 2

Calculate the error $\xi - p(\hat{\omega})$, which is a matrix of shape $(B,2)$ in 2D and $(B,3)$ in 3D. Row $i$ represents the vector that points from the prediction to the target for sample $i$. Its magnitude denotes the distance between the 2 points.  

```python
        # (B, 2) or (B, 3)
        predictions = fk(lengths, angles)
        
        # (B, 2) or (B, 3)
        error = ee_true - predictions
```

The diagram below visualizes this step. The `error` is the yellow dashed vector:

![](../../assets/jit/step2.svg#jit_img)

### Step 3

Since the algorithm is iterative, we need to define a threshold, where if the error is smaller than the threshold, we consider the inverse kinematics problem solved. In my case I use $0.001$. Now we can extract which samples from the batch have not converged yet:

```python
        # Find the samples that have not converged yet
        not_converged_mask = np.linalg.norm(error, axis=1) > tolerance
        
        # If all samples in the batch have converged, we can stop early
        if not not_converged_mask.any(): break
        
        # Filter out the samples that can still be optimized
        # [<=B, N] or [<=B, 2N]
        unknown_angles = angles[not_converged_mask]
```

From now on we will be looking only at them since optimizing the converged ones further is a waste of time.

### Step 4

Calculate the Jacobian matrix $J$ of $p$ with respect to $\hat{\omega}$. It is a $(2,N)$ in 2D or $(3,2N)$ matrix, where the $(i,j)$-th component represents the partial derivative of the $i$-th component of $p(\hat{\omega})$ with respect to the $j$-th component of $\hat{\omega}$. The matrix can be numerically approximated using the definition of the derivative as follows:

$$
J_{ij}=\frac{\partial p_i}{\partial \hat{\omega}_j}\approx\frac{p(\hat{\omega}_j + h)_i - p(\hat{\omega})_i}{h}
$$

Here $\hat{\omega}_j+h$ refers to $\hat{\omega}$ with $h$ added to its $j$-th component, where $h$ is a small positive number that controls the convergence of the algorithm. I use $h=0.005$. Intuitively, the $(i,j)$-th component represents how much angle $j$ changes the $i$-th coordinate of the endeffector position. The Jacobian matrix can be calculated as follows:

#### Step 4.1

Calculate $\hat{\omega}_j + h$:

```python
        perturbed_angles = (

            # [<=B, N, N] or [<=B, 2N, 2N]
            np.tile(unknown_angles[:, np.newaxis], (1, unknown_angles.shape[1], 1)) 
            
            + 
            
            # [N, N] or [2N, 2N]
            H[:unknown_angles.shape[1], :unknown_angles.shape[1]]
        )
```

Here $H$ is the matrix $hI$ defined beforehand as:

```python
    H = h * np.eye(angles.shape[1])
```

The variable `perturbed_angles` is a tensor of shape $(B,N,N)$ for the 2D case or $(B,2N,2N)$ for the 3D case, where the $(i,j)$-th component is the current angles $\hat{\omega}$ for batch sample $i$ with $h$ added to its $j$-th component.

#### Step 4.2

Calculate the perturbed positions $p(\hat{\omega}_j + h)$:

```python
        perturbed_positions = fk(

            # [<=B, 1, N] or [<=B, 1, 2N]
            lengths=lengths[not_converged_mask][:, np.newaxis], 
            
            # [<=B, N, N] or [<=B, 2N, 2N]
            angles=perturbed_angles
        )
```

Here the link lengths are reshaped to $(B,1,N)$ or $(B,1,2N)$ and we can conveniently reuse the forward kinematics function to process this batch of perturbed angles (here we have 2 batch dimensions, which is why we made it so that the forward kinematics functions work with arbitrary batch dimensions). Here we can broadcast the lengths efficiently since we know that the systems in each batch of `perturbed_angles` use the same set of links. The resulting array `perturbed_positions` is of shape $(B,N,2)$ or $(B,2N,3)$.

#### Step 4.3

Calculate the jacobian matrix $J$:

```python
        jacobian_matrix = (

            # [<=B, N, 2] or [<=B, 2N, 3]
            perturbed_positions 

            - 
            
            # [<=B, 1, 2] or [<=B, 1, 3]
            predictions[not_converged_mask][:, np.newaxis]

        ).transpose(0, -1, -2) / h
```

This is done by applying the formula from above and reusing the computed positions $p(\hat{\omega})$ from before. The resulting matrix is transposed to flip the dimensions of the angles and the positions. The resulting `jacobian_matrix` array is of shape $(B, 2, N)$ or $(B, 3, 2N)$, where the $i$-th component is the Jacobian matrix for the $i$-th sample in the batch.

### Step 5

Update the angles $\hat{\omega}$ by taking a step in the direction of $J^+(\xi-p(\hat{\omega}))$ as follows:

$$
\hat{\omega}\gets\hat{\omega} + J^+(\xi-p(\hat{\omega}))
$$

Here $J^+$ is the Moore-Penrose pseudoinverse of $J$ (this is where *Inverse* in *Jacobian Inverse Technique* comes from, not from *Inverse Kinematics*). This expression swerves the $\hat{\omega}$ towards a solution that minimizes the error. This is essentially Newton's method, which is used to iteratively find the roots of a function $f$ given an initial guess $x_0$ by updating the guess as follows:

$$
x_{i+1}\gets x_i-\frac{f(x_i)}{f^\prime(x_i)}
$$

Essentially, $x_{i+1}$ is the point where the tangent line to $f$ at $x_i$ crosses the $x$-axis, which turns out to be a very good heuristic. The method is usually very efficient as it takes only a few iterations to reach a guess $x^*$ such that $f(x^*)\approx0$. For finding the roots of a multidimensional function $g: \mathbb{R^n}\to\mathbb{R^m}$, the update rule of the Newton method generalizes to:

$$
x_{i+1}\gets x_i - J^+g(x_i)
$$

Here $J^+$ is the pseudoinverse of $g$'s Jacobian matrix. Intuitively, "dividing" by a matrix entails multiplying by the inverse. The iterations result in a prediction $x^*$ such that $f(x^*)\approx\mathbf{0}$, i.e. all $m$ components of $f(x_i)$ are simultaneously swerved to 0.

We can prove that the update steps in the Jacobian Inverse Technique and Newton's method are identical:

- Our fundamental goal is to find angles $\omega^*$ such that $p(\omega^*)=\xi$, which is the same as $p(\omega^*)-\xi=\mathbf{0}$.
- Therefore, we can substitute $x=\hat{\omega}$ and $g(\hat{\omega})=\xi-p(\hat{\omega})$ in the Newton's method definition.
- Naturally, the Jacobian of $g$ is equal to the negative Jacobian of $p$. That is because the sum rule allows us to discard $\xi$ since $\xi$ is not dependent on $\hat{\omega}$, which means that we are left with $g^\prime(\hat{\omega})=-p^\prime(\hat{\omega})$.
- Using the property of the pseudoinverse $(kA)^+=k^{-1}A^+$, we can directly infer that $(-A)^+=-A^+$. Therefore, the pseudoinverse of $g$'s Jacobian is the negative pseudoinverse of $p$'s Jacobian. We already calculated the Jacobian of $p$, so we can simply substitute it and flip the sign of the update.

That's it! This is how we can do the update in NumPy:

```python
        delta_angles = (

            # [<=B, N, 2] or [<=B, 2N, 3]
            np.linalg.pinv(jacobian_matrix) 
            
            @
            
            # [<=B, 2, 1] or [<=B, 3, 1]
            error[not_converged_mask][..., np.newaxis]
            
        ).squeeze(-1)
```

The resulting array `delta_angles` is of shape $(B, N)$ or $(B, 2N)$ and we can directly add it to our guess $\hat{\omega}$ as follows:

```python
        angles[not_converged_mask] += delta_angles
```

Now we can simply return the angles once all samples have converged.

```python
    return angles
```

## Interaction

Try the Jacobian Inverse Technique out yourself using the following interactions! You can visualize the change in predictions throughout the iterations and how the algorithm converges to the desired endeffector position.
