# Kolmogorov-Arnold Networks (KAN)

KAN are machine learning models very similar to neural networks. They are composed of stacked layers, where each layer receives an input, transforms it, and passes the output to the next layer in the sequence. Conversely to MLPs that use learnable linear transformation interspersed with fixed activation functions, KAN instead learn the activation functions as well. In this project I implement and derive KAN from scratch.

## B-Splines (Preliminaries)

B-Splines are smooth and differentiable curves. They are composed of $n$ basis elements, parameterized by their degree $p$ and knot vector $t=t_0,t_1,\ldots,t_m$. The definition of basis element $i$ is recursive, with base case

$$
B_{i,0}(x)=\begin{cases}1&\text{if }t_i\leq x<t_{i+1}\\0&\text{otherwise}\end{cases}
$$

And recursive case:
$$
B_{i,p}(x)=\frac{x-t_i}{t_{i+p}-t_i}B_{i,p-1}(x)+\frac{t_{i+p+1}-x}{t_{i+p+1}-t_{i+1}}B_{i+1,p-1}(x)
$$

The spline of degree $p$ itself is a linear combination of its basis elements, i.e., it can be written as follows:

$$
\mathrm{Spline}(x)=\sum_{i=1}^nw_iB_{i,p}(x)
$$

This is how we can define a B-spline over the range $[-1,1]$:

```python
class BSpline:
    
    def __init__(self, p: int = 3, n: int = 7) -> None:
        knots = n+p+1
        grid_size = n-p
        step = 2 / grid_size
        self.t = np.linspace(-1 - p * step, 1 + p * step, knots)
        self.p = p
        self.n = n
```

In this case, the knots are evenly spaced, which is consistent with KAN but is not required. Each basis element is defined over $p+2$ knots, with the first and last knot being the anchors. In total, we would end up with $n$ knots, one for each element, and additional $\left\lfloor \frac{p+1}{2}\right\rfloor$ on each sides. In the code `knots` represents the number of required knots.

Next, we would need exactly the middle $n-p+1$ knots for each element to be included. The variable `grid_size` defines the number of distances for the middle $n-p+1$ knots, i.e. $n-p+1-1=n-p$.

The domain of our spline is $[-1,1]$, which has a total length of 2. To fit exactly $n-p$ knots in this range, they need to be $\frac{2}{n-p}$ apart from each other, which is what the variable `step` represents.

Finally, we can define our knots in the `t` variable by including the $p$ additional knots outside of the predefined range on each side. The following interaction visualizes the B-Spline:

<!-- INTERACTION -->

It can be noticed that changing the weight of a single basis element affects the spline only around the center of that element, which is a key property of B-Splines. In code, the spline is evaluated as follows:

```python
    def _rec_call(self, x: np.ndarray, p: int, i: int, t: np.ndarray) -> np.ndarray:
        if p == 0: return ((t[i] <= x) & (x < t[i+1])).astype(float)
        return (
            (x - t[i]) / (t[i+p] - t[i]) * self._rec_call(x, p-1, i, t)
            +
            (t[i+p+1] - x) / (t[i+p+1] - t[i+1]) * self._rec_call(x, p-1, i+1, t)
        )

    def __call__(self, x: np.ndarray) -> np.ndarray:
        components = [self._rec_call(x, self.p, i, self.t) for i in range(self.n)]
        return np.stack(components)
```

Here the input `x` is a $m$-dimensional vector. The output is the contribution of each basis element at `x`, i.e., a $n\times m$ matrix. The final spline is computed as the linear combination of the contributions according to the weights $w$, which comes later.

The derivative of a B-Spline basis element is defined as follows:

$$
B_{i,k}^\prime(x)=k\left(\frac{B_{i,k-1}(x)}{t_{i+k}-t_i}-\frac{B_{i+1,k-1}(x)}{t_{i+k+1}-t_{i+1}}\right)
$$

In code:

```python
    def derivative(self, x: np.ndarray) -> np.ndarray:
        components = [
            (
                self._rec_call(x, self.p-1, i, self.t) / (self.t[i+self.p] - self.t[i])
                -
                self._rec_call(x, self.p-1, i+1, self.t) / (self.t[i+self.p+1] - self.t[i+1])
            )
            for i in range(self.n)
        ]
        return self.p * np.stack(components)
```

Analogically, the input `x` is a $m$-dimensional vector, and the output is the derivative of each basis element with respect to `x`, i.e., a $n\times m$ matrix.

## KAN Layer

The main layer of the architecture is the KAN alternative to linear layer + non-linear activation combination in MLP. It is defined as follows:

Components:

1. The input vector $\mathbf{x}\in\mathbb{R}^{1\times n}$
2. The output vector $\mathbf{y}\in\mathbb{R}^{1\times m}$
3. The function matrix $\mathbf{\Phi}\in(\mathbb{R}\to\mathbb{R})^{n\times m}$

$$
\mathbf{x}=\begin{bmatrix}x_1 & x_2 & \cdots & x_n\end{bmatrix},\quad\mathbf{y}=\begin{bmatrix}y_1 & y_2 & \cdots & y_m\end{bmatrix},\quad\mathbf{\Phi}=\begin{bmatrix}\phi_{11}(\cdot) & \phi_{12}(\cdot) & \cdots & \phi_{1m}(\cdot)\\\phi_{21}(\cdot) & \phi_{22}(\cdot) & \cdots & \phi_{2m}(\cdot)\\\vdots & \vdots & \ddots & \vdots\\\phi_{n1}(\cdot) & \phi_{n2}(\cdot) & \cdots & \phi_{nm}(\cdot)\\\end{bmatrix}
$$

## Forward Pass

The output of the KAN layer $\mathbf{y}$ is computed as follows:

$$
y_j=\sum_{i=1}^n\phi_{ij}(x_i)
$$

Now we can look at the univariate functions $\phi_{ij}$. They are originally defined as cubic B-Splines with $b$ basis elements as follows:

$$
\phi_{ij}(x)=\sum_{k=1}^bw_{ijk}B_k(x)
$$

Where $w_{ijk}$ are the elements of a learnable weights tensor $\mathbf{W}\in\mathbb{R}^{n\times m\times b}$, which is what the network optimizes. Now we can write the full formula for the output as follows:

$$
y_j=\sum_{i=1}^n\sum_{k=1}^bw_{ijk}B_k(x_i)
$$

And define the output vector $\mathbf{y}$ as

$$
\mathbf{y}=\begin{bmatrix}\sum_{i=1}^n\sum_{k=1}^bw_{i1k}B_k(x_i)&\cdots&\sum_{i=1}^n\sum_{k=1}^bw_{imk}B_k(x_i)\end{bmatrix}
$$

Now we are ready to define the forward pass of a KAN layer:

```python
    def __init__(self, in_features: int, out_features: int, weights: int = 7) -> None:
        
        self.in_features = in_features
        self.out_features = out_features
        self.weights = weights

        self.learning_rate = .05
        self.spline = BSpline(n=weights)
        self.W = np.random.normal(loc=0, scale=.1, size=(in_features, out_features, weights))

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        self.B = self.spline(x)
        y = np.einsum('ijk,ki->j', self.W, self.B)
        return y
```

We initialize each weight $w_{ijk}\sim\mathcal{N}(\mu=0,\sigma=0.1)$. In the code above `self.B` is a $b\times n$ matrix such that `self.B`$_{ki}=B_k(x_i)$. To learn the weights $\mathbf{W}$ we can simply use backpropagation as in regular MLPs. The derivations are shown in the following sections.

## Backward Pass

Given is the derivative of the loss with repect to the output $\frac{\partial \mathcal{L}}{\partial \mathbf{y}}\in\mathbb{R}^{m\times 1}$, defined as:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{y}}=\begin{bmatrix}\frac{\partial \mathcal{L}}{\partial y_1} & \frac{\partial \mathcal{L}}{\partial y_2} & \cdots & \frac{\partial \mathcal{L}}{\partial y_m}\end{bmatrix}
$$

Using this, we need to update the weights $\mathbf{W}$ and propagate the loss by passing the derivative of the loss with respect to the input $\mathbf{x}$ to the previous layer.

### Finding the derivative of the loss with respect to the weights

The derivative of the loss with respect to the weights $\frac{\partial \mathcal{L}}{\partial \mathbf{W}}\in\mathbb{R}^{n\times m\times b}$ is defined as:

$$
\frac{\partial\mathcal{L}}{\partial w_{ijk}}=\frac{\partial\mathcal{L}}{\partial y_j}B_k(x_i)
$$

<!-- derivation -->

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{W}}=\begin{bmatrix}\begin{bmatrix}\frac{\partial \mathcal{L}}{\partial w_{111}} & \cdots & \frac{\partial \mathcal{L}}{\partial w_{11b}}\\\vdots&\ddots&\vdots\\\frac{\partial \mathcal{L}}{\partial w_{1m1}} & \cdots & \frac{\partial \mathcal{L}}{\partial w_{1mb}}\end{bmatrix}\\\vdots\\\begin{bmatrix}\frac{\partial \mathcal{L}}{\partial w_{n11}} & \cdots & \frac{\partial \mathcal{L}}{\partial w_{n1b}}\\\vdots&\ddots&\vdots\\\frac{\partial \mathcal{L}}{\partial w_{nm1}} & \cdots & \frac{\partial \mathcal{L}}{\partial w_{nmb}}\end{bmatrix}\end{bmatrix}
$$

Applying the chain rule to each component gives:

$$
\frac{\partial \mathcal{L}}{\partial w_{ijk}}=\frac{\partial \mathcal{L}}{\partial \mathbf{y}}\frac{\partial\mathbf{y}}{\partial w_{ijk}}
$$

Since $\frac{\partial \mathcal{L}}{\partial \mathbf{y}}$ is given, we only need to find $\frac{\partial\mathbf{y}}{\partial w_{ijk}}\in\mathbb{R}^{m\times 1}$, defined as:

$$
\frac{\partial\mathbf{y}}{\partial w_{ijk}}=\begin{bmatrix}\frac{\partial y_1}{\partial w_{ijk}}\\\vdots\\\frac{\partial y_m}{\partial w_{ijk}}\end{bmatrix}
$$

From the formula for the forward pass we can see that:

$$
\frac{\partial y_l}{\partial w_{ijk}}=\begin{cases}B_k(x_i)&\text{if }l=j\\0&\text{otherwise}\end{cases}
$$

Therefore,

$$
\frac{\partial \mathcal{L}}{\partial w_{ijk}}=\frac{\partial \mathcal{L}}{\partial \mathbf{y}}\frac{\partial\mathbf{y}}{\partial w_{ijk}}=\begin{bmatrix}\frac{\partial \mathcal{L}}{\partial y_1} & \cdots & \frac{\partial \mathcal{L}}{\partial y_m}\end{bmatrix}\begin{bmatrix}0\\\vdots\\B_k(x_i)\\\vdots\\0\end{bmatrix}=\frac{\partial \mathcal{L}}{\partial y_j}B_k(x_i)
$$

<!-- derivation -->

Now we can simply update the weights with gradient descent

$$
\mathbf{W}\gets\mathbf{W}-\alpha\frac{\partial \mathcal{L}}{\partial \mathbf{W}}
$$

### Finding the derivative of the loss with respect to the input

The goal is to find the derivative of the loss with respect to the input $\frac{\partial \mathcal{L}}{\partial \mathbf{x}}\in\mathbb{R}^{1\times n}$, defined as:

$$
\frac{\partial \mathcal{L}}{\partial x_i}=\sum_{j=1}^m\frac{\partial \mathcal{L}}{\partial y_j}\sum_{k=1}^bw_{ijk}B_k^\prime(x_i)
$$

<!-- derivation -->

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{x}}=\begin{bmatrix}\frac{\partial \mathcal{L}}{\partial x_1} & \frac{\partial \mathcal{L}}{\partial x_2} & \cdots & \frac{\partial \mathcal{L}}{\partial x_n}\end{bmatrix}
$$

Applying the chain rule to each component gives:

$$
\frac{\partial \mathcal{L}}{\partial x_i}=\frac{\partial \mathcal{L}}{\partial \mathbf{y}}\frac{\partial\mathbf{y}}{\partial x_i}
$$

Now we only need to find $\frac{\partial\mathbf{y}}{\partial x_i}\in\mathbb{R}^{n\times 1}$, defined as:

$$
\frac{\partial\mathbf{y}}{\partial x_i}=\begin{bmatrix}\frac{\partial y_1}{\partial x_i}\\\vdots\\\frac{\partial y_m}{\partial x_i}\end{bmatrix}
$$

From the formula for the forward pass we can see that:

$$
\frac{\partial y_j}{\partial x_i}=\sum_{k=1}^bw_{ijk}B_k^\prime(x_i)
$$

Therefore,

$$
\frac{\partial \mathcal{L}}{\partial x_i}=\frac{\partial \mathcal{L}}{\partial \mathbf{y}}\frac{\partial\mathbf{y}}{\partial x_i}=\begin{bmatrix}\frac{\partial \mathcal{L}}{\partial y_1} & \cdots & \frac{\partial \mathcal{L}}{\partial y_m}\end{bmatrix}\begin{bmatrix}\sum_{k=1}^bw_{i1k}B_k^\prime(x_i)\\\vdots\\\sum_{k=1}^bw_{imk}B_k^\prime(x_i)\end{bmatrix}=\sum_{j=1}^m\frac{\partial \mathcal{L}}{\partial y_j}\sum_{k=1}^bw_{ijk}B_k^\prime(x_i)
$$

<!-- derivation -->

Since the input to a layer is the output of the previous layer, we can now simply pass $\frac{\partial \mathcal{L}}{\partial \mathbf{x}}$ as $\frac{\partial \mathcal{L}}{\partial \mathbf{y}}$ for the previous layer. And we are done! In code this is done as follows:

```python
    def backward(self, dEdY: np.ndarray) -> np.ndarray:
        dEdW = np.einsum('j,ki->ijk', dEdY, self.B)
        self.W -= self.learning_rate * dEdW
        B_prime = self.spline.derivative(self.x)
        dEdX = np.einsum('j,ijk,ki->i', dEdY, self.W, B_prime)
        return dEdX
```

Here we can conveniently use `np.einsum`, which pretty much directly translates the mathematical expressions into code. This allows us to do complex operations on tensors in a readable, simple, and efficient manner.


## Implementation details

The B-splines are defined only over a specific range (in our case $[-1, 1]$) even though the output of a FC layer is unbounded. To keep the implementation simple, I apply the $\tanh$ function element-wise to the output of each layer, ensuring that the output is in the range $[-1, 1]$. The authors also propose that as a solution to this problem, however, they develop a procedure to dynamically extend the grid such that it matches the input shape.

The $\tanh$ layer is defined as follows:

```python
class Tanh:
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        return np.tanh(x)
    def backward(self, dEdY: np.ndarray) -> np.ndarray:
        dEdX = dEdY * (1 - np.tanh(self.x) ** 2)
        return dEdX
```

The loss function will be the standard mean squared error, i.e. $\mathrm{MSE}$, defined as follows:

```python
class MSE:    
    def forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return ((y_true - y_pred) ** 2).mean()

    def backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return 2 * (y_pred - y_true) / y_true.size
```

Now we have all the components to define a fully functioning KAN. Again, other than the KAN layer, the rest is identical to a neural network. The entire model with its training loop can be defined as follows:

```python
class KAN:
    
    def __init__(self) -> None:

        self.layers = [
            Layer(1, 2),
            Tanh(),
            Layer(2, 2),
            Tanh(),
            Layer(2, 1)
        ]
        self.loss = MSE()

    def predict(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 500):

        for epoch in tqdm(range(epochs)):

            loss = 0
            for x, y_true in zip(X, y):
                y_pred = self.predict(x)
                loss += self.loss.forward(y_true, y_pred)
                error = self.loss.backward(y_true, y_pred)
                for layer in reversed(self.layers): error = layer.backward(error)
```

## Interaction

The following plot shows the results of training a KAN with 1 hidden layer with 2 neurons for 100 epochs on 20 samples of the following function.

$$
f(x)=\frac{1}{2}\sin(4x)\exp(-x-1)
$$

Unfortunately, the model training is slow, mainly due to the native python implementation of B-Splines. One can achieve way faster training using the analogical B-Spline implementation from [scipy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.BSpline.html).

<!-- INTERACTION -->

KAN promote explainability since one can easily visualize the splines on the edges. Try clicking the edges on the following graph representation of the fully trained model to see the splines!
