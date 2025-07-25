# Neural Networks

Neural networks are machine learning models composed of stacked *layers*. Each layer receives an input and transforms it into an output. The transformation in each layer must be differentiable, which enables the layers' parameters to be trained using gradient descent. The 2 most basic types of layers are linear layers and activation layers. Given an input vector $\mathbf{x}\in\mathbb{R}^{n}$, the linear layer outputs a linear transformation of it, $\mathbf{xW}$, where $\mathbf{W}\in\mathbb{R}^{n\times m}$ are learnable weights. The activation later applies a non-linear univariate activation function $\sigma:\mathbb{R}\to\mathbb{R}$ element-wise to the input, which is necessary to make the whole model non-linear. Most commonly, neural network architectures involve a stack of linear layers interspersed with activation layers. That way, the whole network can be represented as a single mathematical function $h:\mathbb{R}^n\to\mathbb{R}^m$ as follows:

$$
h(\mathbf{x})=\left(\mathbf{W}^{(L)} \circ \sigma \circ \mathbf{W}^{(L-1)} \circ \cdots \sigma \circ \mathbf{W}^{(1)}\right)(\mathbf{x}).
$$

Now it can be seen that if the activation layers were omitted, $h$ would become the following:

$$
h(\mathbf{x})=\mathbf{x}\left(\prod_{l=1}^LW^{(L-l)}\right),
$$

which is the same as we had a single linear layer with weights $\left(\prod_{l=1}^LW^{(L-l)}\right)$. That is why a neural network with an arbitraty number of linear layers and no non-linear activation functions will never theoretically outpreform a regular linear regression.

The output of the model $\hat{\mathbf{y}}=h(\mathbf{x})$ is its predictions for the outputs. To quantitatively compare them with the ground truth labels $\mathbf{y}\in\mathbb{R}^m$, one must define a differentiable loss function $\mathcal{L}:\mathbb{R}^m\times\mathbb{R}^m\to\mathbb{R}$. The larger $\mathcal{L}$ is, the larger the difference between the prediction and ground truths is. Therefore, the model aims to minimize the loss. The parameters are updated using the backpropagation algorithm, which entails taking a step in the oppsite direction of the loss function's gradient. Naturally, the gradient points in the direction of fastest increase, therefore, the fastest decrease is in the oppsite direction. Similar logic is applied in [Newton's method](https://en.wikipedia.org/wiki/Newton%27s_method) for function approximation. Since the output of layer $i$ is the input of layer $i+1$, we can apply the chain rule to update the parameters of each layer simultenously. The following outlines the mathematical properties of this algorithm for each type of layer and the loss function from an algebraic perspective rather than the common (in my opinion, less intuitive), geometric perspective.

Remark on notation: The derivative of a function $f:\mathbb{R}^n\to\mathbb{R}^m$ with respect to its input $\mathbf{x}$ is a $m\times n$ matrix, where the $(ij)$-th element is $\frac{\partial f_j}{\partial x_i}$, i.e. the derivative of the $j$-th component of $f$ with respect to the $i$-th component of $\mathbf{x}$.

We will be using the following blueprint for creating the 2 types of layers:

```python
class Layer:
    
    def __init__(self, lr: float = .01) -> None:

        # To store the input x        
        self.x: np.ndarray = None
        
        # Learning rate
        self.lr = lr

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Define the forward pass"""

        raise NotImplementedError
    
    def backward(self, dEdY: np.ndarray) -> np.ndarray:
        """Backpropagation"""

        raise NotImplementedError
```

## Linear Layer

As mentioned above, the linear layer outputs a linear transformation of its input.

Components:

1. Learnable weight matrix $\mathbf{W}\in\mathbb{R}^{n\times m}$
2. Input vector $\mathbf{x}\in\mathbb{R}^{1\times n}$
3. Bias vector $\mathbf{b}\in\mathbb{R}^{1\times m}$
4. Output vector $\mathbf{y}\in\mathbb{R}^{1\times m}$

$$
\mathbf{W}= \begin{bmatrix} w_{11} & w_{12} & \cdots & w_{1m} \\ w_{21} & w_{22} & \cdots & w_{2m} \\ \vdots & \vdots & \ddots & \vdots \\ w_{n1} & w_{n2} & \cdots & w_{nm} \end{bmatrix}, \quad \mathbf{b} = \begin{bmatrix} b_1 & b_2 & \cdots & b_m \end{bmatrix}, \quad \mathbf{x}=\begin{bmatrix}x_1 & x_2 & \cdots & x_n\end{bmatrix}, \quad \mathbf{y}=\begin{bmatrix}y_1 & y_2 & \cdots & y_m \end{bmatrix}
$$

### Forward pass

The output $\mathbf{y}$ is calculated as $\mathbf{y}=\mathbf{xW + b}$, i.e. as follows:

$$
y_i=\sum_{j=1}^nx_jw_{ji}+b_i
$$

Or in full:

$$
\mathbf{y}=\begin{bmatrix}\sum_{j=1}^nx_jw_{j1}+b_1 & \sum_{j=1}^nx_jw_{j2}+b_2 & \cdots & \sum_{j=1}^nx_jw_{jm}+b_m\end{bmatrix}
$$

In code this is done as follows:

```python
class Linear(Layer):
    
    def __init__(self, n: int, m: int, lr: float = .01) -> None:
        
        super().__init__(lr)

        # weights W
        self.W = np.random.randn(n, m)
        
        # bias b
        self.b = np.random.randn(1, m)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass: y = xW + b"""

        x = np.atleast_2d(x)
        self.x = x
        y = x @ self.W + self.b
        return y
```

### Backward pass

The input is the derivative of the loss function $\mathcal{L}$ with respect to each component of the output $\mathbf{y}$, i.e. $\frac{\partial \mathcal{L}}{\partial \mathbf{y}} \in\mathbb{R}^{1\times m}$, which is defined as:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{y}} = \begin{bmatrix} \frac{\partial \mathcal{L}}{\partial y_1} & \frac{\partial \mathcal{L}}{\partial y_2} & \cdots & \frac{\partial \mathcal{L}}{\partial y_m} \end{bmatrix}
$$

Using this, we need to find the derivative of the error with respect to the weights, the bias, and the input. The first 2 are used to update the model's parameters, while the latter is the derivative of the error with respect to the outputs of the previous layer, so it is directly passed to it and the same operations are repeated. This can be applied recursively to update all layers.

#### The derivative of the error with respect to the weights

The derivative of the loss $\mathcal{L}$ with respect to the weights $\mathbf{W}$, i.e. $\frac{\partial \mathcal{L}}{\partial \mathbf{W}}\in\mathbb{R}^{n\times m}$ is defined as:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{W}}=\mathbf{x}^\top\frac{\partial\mathcal{L}}{\partial\mathbf{y}}
$$

<!-- derivation -->

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{W}} = \begin{bmatrix} \frac{\partial \mathcal{L}}{\partial w_{11}} & \cdots & \frac{\partial \mathcal{L}}{\partial w_{1m}} \\ \vdots & \ddots & \vdots \\ \frac{\partial \mathcal{L}}{\partial w_{n1}} & \cdots & \frac{\partial \mathcal{L}}{\partial w_{nm}}\end{bmatrix}
$$

We can apply the chain rule to each component:

$$
\frac{\partial \mathcal{L}}{\partial w_{ji}} = \frac{\partial \mathcal{L}}{\partial \mathbf{y}}\frac{\partial \mathbf{y}}{\partial w_{ji}}
$$

Since $\frac{\partial \mathcal{L}}{\partial \mathbf{y}}$ is known, we only need to find $\frac{\partial \mathbf{y}}{\partial w_{ji}}\in\mathbb{R}^{m\times 1}$ (I use $ji$ since the relationship with the forward pass can be seen better), given as:

$$
\frac{\partial \mathbf{y}}{\partial w_{ji}}=\begin{bmatrix}\frac{\partial y_1}{\partial w_{ji}} \\ \vdots \\ \frac{\partial y_m}{\partial w_{ji}} \end{bmatrix}
$$

From the forward pass formula we can see that

$$
\frac{\partial y_k}{\partial w_{ji}}=\begin{cases} x_j & \text{if } i = k \\ 0 & \text{otherwise}\end{cases}
$$

Therefore

$$
\frac{\partial \mathcal{L}}{\partial w_{ji}}=\frac{\partial \mathcal{L}}{\partial \mathbf{y}}\frac{\partial \mathbf{y}}{\partial w_{ji}}=\begin{bmatrix} \frac{\partial \mathcal{L}}{\partial y_1} & \cdots & \frac{\partial \mathcal{L}}{\partial y_m} \end{bmatrix}\begin{bmatrix}0 \\ \vdots \\ x_j \\ \vdots \\ 0 \end{bmatrix} = \frac{\partial \mathcal{L}}{\partial y_i}x_j
$$

Substituting back to $\frac{\partial \mathcal{L}}{\partial \mathbf{W}}$, we get

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{W}} = \begin{bmatrix} \frac{\partial \mathcal{L}}{\partial y_1}x_1 & \cdots & \frac{\partial \mathcal{L}}{\partial y_m}x_1 \\ \vdots & \ddots & \vdots \\ \frac{\partial \mathcal{L}}{\partial y_1}x_n & \cdots & \frac{\partial \mathcal{L}}{\partial y_m}x_n \end{bmatrix} = \begin{bmatrix} x_1 \\ \vdots \\ x_n \end{bmatrix}\begin{bmatrix} \frac{\partial \mathcal{L}}{\partial y_1} & \cdots & \frac{\partial \mathcal{L}}{\partial y_m} \end{bmatrix}=\mathbf{x}^\top\frac{\partial \mathcal{L}}{\partial \mathbf{y}}
$$

<!-- derivation -->

Now we can update the weights by taking a small step in the opposite direction as follows:

$$
\mathbf{W}\gets\mathbf{W}-\alpha\frac{\partial \mathcal{L}}{\partial \mathbf{W}}
$$

#### The derivative of the error with respect to the bias

The derivative of the loss $\mathcal{L}$ with respect to the bias $\mathbf{b}$, i.e. $\frac{\partial \mathcal{L}}{\partial \mathbf{b}}\in\mathbb{R}^{1\times m}$ is defined as

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{b}}=\frac{\partial \mathcal{L}}{\partial \mathbf{y}}
$$

<!-- derivation -->

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{b}} = \begin{bmatrix} \frac{\partial \mathcal{L}}{\partial b_1} & \frac{\partial \mathcal{L}}{\partial b_2} & \cdots & \frac{\partial \mathcal{L}}{\partial b_m} \end{bmatrix}
$$

Applying the chain rule to each component:

$$
\frac{\partial \mathcal{L}}{\partial b_i}=\frac{\partial \mathcal{L}}{\partial \mathbf{y}}\frac{\partial \mathbf{y}}{\partial b_i}
$$

Analogically, we only need to find $\frac{\partial \mathbf{y}}{\partial b_i}\in\mathbb{R}^{1\times m}$, given as:

$$
\frac{\partial \mathbf{y}}{\partial b_i}=\begin{bmatrix}\frac{\partial y_1}{\partial b_i} \\ \vdots \\ \frac{\partial y_m}{\partial b_i} \end{bmatrix}
$$

Again, from the forward pass formula we can see that 

$$
\frac{\partial y_k}{\partial b_i} = \begin{cases} 1 & \text{if } i = k \\ 0 & \text{otherwise} \end{cases}
$$

Therefore

$$
\frac{\partial \mathcal{L}}{\partial b_i}=\frac{\partial \mathcal{L}}{\partial \mathbf{y}}\frac{\partial \mathbf{y}}{\partial b_i}=\begin{bmatrix} \frac{\partial \mathcal{L}}{\partial y_1} & \cdots & \frac{\partial \mathcal{L}}{\partial y_m} \end{bmatrix}\begin{bmatrix}0 \\ \vdots \\ 1 \\ \vdots \\ 0 \end{bmatrix} = \frac{\partial \mathcal{L}}{\partial y_i}
$$

Substituting back to $\frac{\partial \mathcal{L}}{\partial \mathbf{b}}$, we get

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{b}} = \begin{bmatrix} \frac{\partial \mathcal{L}}{\partial y_1} & \cdots & \frac{\partial \mathcal{L}}{\partial y_m} \end{bmatrix} = \frac{\partial \mathcal{L}}{\partial \mathbf{y}}
$$

<!-- derivation -->

Now we can update the bias by taking a small step in the opposite direction as follows:

$$
\mathbf{b}\gets\mathbf{b}-\alpha\frac{\partial \mathcal{L}}{\partial \mathbf{b}}
$$

#### The derivative of the error with respect to the input

The derivative of the loss $\mathcal{L}$ with respect to the input $\mathbf{x}$, i.e. $\frac{\partial \mathcal{L}}{\partial \mathbf{x}}\in\mathbb{R}^{1\times n}$ is defined as

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{x}}=\frac{\partial\mathcal{L}}{\partial\mathbf{y}}\mathbf{W}^\top
$$

<!-- derivation -->

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{x}} = \begin{bmatrix} \frac{\partial \mathcal{L}}{\partial x_1} & \cdots & \frac{\partial \mathcal{L}}{\partial x_n} \end{bmatrix}
$$

Applying the chain rule to each component:

$$
\frac{\partial \mathcal{L}}{\partial x_j}=\frac{\partial \mathcal{L}}{\partial \mathbf{y}}\frac{\partial \mathbf{y}}{\partial x_j}
$$

Analogically, we only need to find $\frac{\partial \mathbf{y}}{\partial x_j}\in\mathbb{R}^{1\times n}$, given as:

$$
\frac{\partial \mathbf{y}}{\partial x_j}=\begin{bmatrix}\frac{\partial y_1}{\partial x_j} \\ \vdots \\ \frac{\partial y_m}{\partial x_j} \end{bmatrix}
$$

Again, from the forward pass formula we can see that

$$
\frac{\partial y_i}{\partial x_j} = w_{ji}
$$

Therefore

$$
\frac{\partial \mathcal{L}}{\partial x_j}=\frac{\partial \mathcal{L}}{\partial \mathbf{y}}\frac{\partial \mathbf{y}}{\partial x_j} = \begin{bmatrix} \frac{\partial \mathcal{L}}{\partial y_1} & \cdots & \frac{\partial \mathcal{L}}{\partial y_m} \end{bmatrix}\begin{bmatrix}w_{j1} \\ \vdots \\ w_{jm} \end{bmatrix} = \sum_{i=1}^m \frac{\partial \mathcal{L}}{\partial y_i}w_{ji}
$$

Substituting back to $\frac{\partial \mathcal{L}}{\partial \mathbf{x}}$, we get

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{x}} = \begin{bmatrix} \sum_{i=1}^m\frac{\partial \mathcal{L}}{\partial y_i}w_{1i} & \cdots & \sum_{i=1}^m\frac{\partial \mathcal{L}}{\partial y_i}w_{ni} \end{bmatrix} = \begin{bmatrix} \frac{\partial \mathcal{L}}{\partial y_1} & \cdots \frac{\partial \mathcal{L}}{\partial y_m} \end{bmatrix}\begin{bmatrix}w_{11} & \cdots & w_{n1} \\ \vdots & \ddots & \vdots \\ w_{1m} & \cdots & w_{nm} \end{bmatrix}=\frac{\partial \mathcal{L}}{\partial \mathbf{y}}\mathbf{W}^\top
$$

<!-- derivation -->

Now that we know $\frac{\partial \mathcal{L}}{\partial \mathbf{x}}$, we also know $\frac{\partial \mathcal{L}}{\partial \mathbf{y}}$ of the previous layer since the current layer's input *is* the previous layer's output. That way we can continue training!

In code:

```python
    def backward(self, dEdy: np.ndarray) -> np.ndarray:
        """Update the parameters and propagate the error"""

        # Calculate the derivative of the error with respect to the weights 
        dEdW = self.x.T @ dEdy

        # The error with respect to the bias
        dEdb = dEdy

        # The error with respect to the input
        dEdx = dEdy @ self.W.T

        # Update the parameters using gradient descent
        self.W -= self.lr * dEdW
        self.b -= self.lr * dEdb

        # Propagate the error
        return dEdx
```

## Activation Layer

The activation layer applies a non-linear function to each sample in the input. It has no learnable parameters.

Definitions:

1. Input vector $\mathbf{x}\in\mathbb{R}^{1\times n}$
2. Output vector $\mathbf{y}\in\mathbb{R}^{1\times n}$
3. Non-linear activation function $f:\mathbb{R}\rightarrow \mathbb{R}$

### Forward pass

The output $\mathbf{y}$ is calculated as $y_i=f(x_i)$, i.e.

$$
\mathbf{y} = \begin{bmatrix} f(x_1) & f(x_2) & \cdots & f(x_n) \end{bmatrix}
$$

In code:

```python
class Activation(Layer):

    def __init__(self, f: Callable[[np.ndarray], np.ndarray], f_prime: Callable[[np.ndarray], np.ndarray]) -> None:
        
        super().__init__()

        # Activation function f
        self.f = f

        # Derivative of the activation function f'
        self.f_prime = f_prime
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass: y = [f(x1) ... f(xn)]"""

        self.x = x
        y = self.f(x)
        return y
```

### Backward pass

Since the activation layer has no learnable parameters, we only need to find the derivative of the loss $\mathcal{L}$ with respect to the input $\mathbf{x}$, i.e. $\frac{\partial \mathcal{L}}{\partial \mathbf{x}}\in\mathbb{R}^{1\times n}$, and pass it to the previous layer. It is defined as

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{x}}=\frac{\partial\mathcal{L}}{\partial\mathbf{y}}\odot f^\prime(\mathbf{x})
$$

<!-- derivation -->

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{x}} = \begin{bmatrix} \frac{\partial \mathcal{L}}{\partial x_1} & \frac{\partial \mathcal{L}}{\partial x_2} & \cdots & \frac{\partial \mathcal{L}}{\partial x_n} \end{bmatrix}
$$

Applying the chain rule to each component gives

$$
\frac{\partial \mathcal{L}}{\partial x_i}=\frac{\partial \mathcal{L}}{\partial \mathbf{y}}\frac{\partial \mathbf{y}}{\partial x_i}
$$

Since $\frac{\partial \mathcal{L}}{\partial \mathbf{y}}$ is known, we only need to find $\frac{\partial \mathbf{y}}{\partial x_i}$, which is given as:

$$
\frac{\partial \mathbf{y}}{\partial x_i} = \begin{bmatrix} \frac{\partial y_1}{\partial x_i} \\ \vdots \\ \frac{\partial y_n}{\partial x_i} \end{bmatrix}
$$

From the forward pass formula we can see that

$$
\frac{\partial y_j}{\partial x_i} = \begin{cases} f^\prime(x_i) & \text{if } j = i \\ 0 & \text{otherwise} \end{cases}
$$

Therefore

$$
\frac{\partial \mathcal{L}}{\partial x_i}=\frac{\partial \mathcal{L}}{\partial \mathbf{y}}\frac{\partial \mathbf{y}}{\partial x_i}=\begin{bmatrix} \frac{\partial \mathcal{L}}{\partial y_1} & \cdots & \frac{\partial \mathcal{L}}{\partial y_m} \end{bmatrix}\begin{bmatrix}0 \\ \vdots \\ f^\prime(x_i) \\ \vdots \\ 0 \end{bmatrix} = \frac{\partial \mathcal{L}}{\partial y_i}f^\prime(x_i)
$$

Substituting back to $\frac{\partial \mathcal{L}}{\partial \mathbf{x}}$, we get

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{x}} = \begin{bmatrix} \frac{\partial \mathcal{L}}{\partial y_1}f^\prime(x_i) & \cdots & \frac{\partial \mathcal{L}}{\partial y_n}f^\prime(x_n) \end{bmatrix} = \frac{\partial \mathcal{L}}{\partial \mathbf{y}}\odot f^\prime(\mathbf{x})
$$

<!-- derivation -->

Now, $\frac{\partial \mathcal{L}}{\partial \mathbf{x}}$ is passed to the previous layer, similarly to the linear layer.

In code:

```python
    def backward(self, dEdy: np.ndarray) -> np.ndarray:
        """Propagate the error"""

        dEdx = dEdy * self.f_prime(self.x)
        return dEdx
```

## Loss Function

The loss function compares the model's predictions and the ground truth labels.

Definitions:

1. Predictions vector $\hat{\mathbf{y}}\in\mathbb{R}^{1\times n}$
2. Ground truth vector $\mathbf{y}\in\mathbb{R}^{1\times n}$

This project focuses on the Mean Squared Error loss, $\text{MSE}:\mathbb{R}^{1\times n}\times\mathbb{R}^{1\times n}\rightarrow \mathbb{R}$, defined as

$$
\text{MSE}(\mathbf{y},\hat{\mathbf{y}})=\frac{1}{n}\sum_{i=1}^n(y_i-\hat{y}_i)^2
$$

Its derivative with respect to the predictions $\hat{\mathbf{y}}$, is defined as

$$
\frac{\partial\mathcal{L}}{\partial\hat{\mathbf{y}}}=\frac{2}{n}\left(\hat{\mathbf{y}}-\mathbf{y}\right)
$$

<!-- derivation -->

$$
\frac{\partial \mathcal{L}}{\partial \hat{\mathbf{y}}} = \begin{bmatrix} \frac{\partial \mathcal{L}}{\partial \hat{y}_1} & \cdots & \frac{\partial \mathcal{L}}{\partial \hat{y}_n} \end{bmatrix}
$$

Now we can look at each component

$$
\frac{\partial \mathcal{L}}{\partial \hat{y}_j}=\frac{\partial}{\partial \hat{y}_j}\left(\frac{1}{n}\sum_{i=1}^n(y_i-\hat{y}_i)^2\right) = \frac{1}{n}\sum_{i=1}^n\frac{\partial}{\partial \hat{y}_j}(y_i-\hat{y}_i)^2
$$

Above I extracted the common factor $\frac{1}{n}$ and pushed $\frac{\partial}{\partial y_j}$ into the sum (which is fine due to the linearity of the differentiation operation). Now we know that $\frac{\partial}{\partial \hat{y}_j}(y_i-\hat{y}_i)^2\neq 0$ iff $j=i$. Therefore, there is only 1 non-zero term in the sum, meaning that we can simplify as follows:

$$
\frac{\partial \mathcal{L}}{\partial \hat{y}_j}=\frac{1}{n}\frac{\partial}{\partial \hat{y}_j}(y_j-\hat{y}_j)^2=\frac{2}{n}(\hat{y}_j-y_j)
$$

Substituting back to $\frac{\partial \mathcal{L}}{\partial \hat{\mathbf{y}}}$, we get

$$
\frac{\partial \mathcal{L}}{\partial \hat{\mathbf{y}}} = \begin{bmatrix} \frac{2}{n}(\hat{y}_1-y_1) & \cdots & \frac{2}{n}(\hat{y}_n-y_n) \end{bmatrix} = \frac{2}{n}(\hat{\mathbf{y}}-\mathbf{y})
$$

<!-- derivation -->

The result is $\frac{\partial \mathcal{L}}{\partial \mathbf{y}}$ of the final layer.

In code:

```python
def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return ((y_true - y_pred) ** 2).mean()

def mse_prime(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return 2 * (y_pred - y_true) / y_true.size
```

## Putting it all together

Some common activation functions are $\text{ReLU}$, defined as $\text{ReLU}(x)=\max(0,x)$ and the hyperbolic tangent $\tanh$. We can define them and their derivatives as follows:

```python
def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)

def tanh_prime(x: np.ndarray) -> np.ndarray:
    return 1 - np.tanh(x) ** 2

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)

def relu_prime(x: np.ndarray) -> np.ndarray:
    return x > 0
```

Now we can make our simple neural network!

```python
class NN:
    
    def __init__(self, activation: str = 'tanh', n_layers: int = 1, size: int = 16, lr: float = .01) -> None:

        f, f_prime = {'relu': (relu, relu_prime), 'tanh': (tanh, tanh_prime)}[activation]

        self.layers: List[Layer] = [
            Linear(1, size, lr),
            Activation(f, f_prime)
        ]

        for i in range(n_layers-1):
            self.layers.extend([
                Linear(size, size, lr), 
                Activation(f, f_prime)
            ])

        self.layers.append(Linear(size, 1))
        
        self.loss = mse
        self.loss_prime = mse_prime
```

Here `n_layers` is the number of hidden layers, excluding the input and output layer, `size` is the number of units per hidden layer, and `lr` is the learning rate ($\alpha$) for all layers. For visualization purposes, the network will be trained on univariate functions. Therefore, we can directly hard code it for simplicity that the network expects a scalar input and produces a scalar output.

The forward pass is simply the input $\mathbf{x}$ passed to the chain of all layers. It can be implemented as follows:

```python
    def predict(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x
```

The entire training loop can be put in a single simple function as follows:

```python
    def fit(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        epochs: int = 1_000
    ) -> None:
        """Train the NN"""

        for epoch in tqdm(range(epochs)):

            # Accumulate the loss (1 value per sample)
            loss = 0

            # Perform an optimization step for each sample
            for x, y_true in zip(X, y):

                # Do the forward pass
                y_pred = self.predict(x)
                
                # Calculate the loss
                loss += self.loss(y_true, y_pred)
                
                # Calculate the gradient of the loss with respect to the outputs
                error = self.loss_prime(y_true, y_pred)

                # Sequentially update the layers with backpropagation
                for layer in reversed(self.layers): error = layer.backward(error)
            
            print(f'Epoch: {epoch} | Loss: {loss / len(X)}')
```

Here `X` is a matrix where each row is an input sample and each column is a feature, and `y` is a vector representing the ground truth labels for each sample. Also, `epochs` is the number of times the model sees the entire dataset. It can be seen that the model processes and trains on each input independently. In practice, it is useful to use batches, where the model can process multiple outputs at once using efficient vectorized operations. Then, the propagated gradient is averaged over the entire batch. For simplicity (especially the mathy part), this project implements only the single-input gradient descent.

# Interaction

Try out the neural network itself with the following interaction!
