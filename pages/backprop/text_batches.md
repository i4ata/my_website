# Backpropagation

In this project I implement neural networks from scratch from a purely algebraic point of view. Neural networks are machine learning models composed of stacked *layers*. Each layer receives an input and transforms it into an output. The transformation in each layer must be differentiable, which enables the layers' parameters to be trained using gradient descent. The 2 most basic types of layers are linear layers and activation layers. Given an input vector $\mathbf{x}\in\mathbb{R}^{n}$, the linear layer outputs a linear transformation of it, $\mathbf{xW}$, where $\mathbf{W}\in\mathbb{R}^{n\times m}$ are learnable weights. The activation later applies a non-linear univariate activation function $\sigma:\mathbb{R}\to\mathbb{R}$ element-wise to the input, which is necessary to make the whole model non-linear. Most commonly, neural network architectures involve a stack of linear layers interspersed with activation layers. That way, the whole network can be represented as a single mathematical function $h:\mathbb{R}^n\to\mathbb{R}^m$ as follows:

$$
h(\mathbf{x})=\left(\mathbf{W}^{(L)} \circ \sigma \circ \mathbf{W}^{(L-1)} \circ \cdots \sigma \circ \mathbf{W}^{(1)}\right)(\mathbf{x}).
$$

Now it can be seen that if the activation layers were omitted, $h$ would become the following:

$$
h(\mathbf{x})=\mathbf{x}\left(\prod_{l=1}^LW^{(L-l)}\right),
$$

which is the same as we had a single linear layer with weights $\left(\prod_{l=1}^LW^{(L-l)}\right)$. That is why a neural network with an arbitraty number of linear layers and no non-linear activation functions will never theoretically outpreform a regular linear regression.

The output of the model $\hat{\mathbf{y}}=h(\mathbf{x})$ is its predictions for the inputs. To quantitatively compare them with the ground truth labels $\mathbf{y}\in\mathbb{R}^m$, one must define a differentiable loss function $\mathcal{L}:\mathbb{R}^m\times\mathbb{R}^m\to\mathbb{R}$. The larger $\mathcal{L}$ is, the larger the difference between the prediction and ground truths is. Therefore, the model aims to minimize the loss. The parameters are updated using the backpropagation algorithm, which entails taking a step in the oppsite direction of the loss function's gradient. Naturally, the gradient points in the direction of fastest increase, therefore, the fastest decrease is in the oppsite direction. Similar logic is applied in [Newton's method](https://en.wikipedia.org/wiki/Newton%27s_method) for function approximation. Since the output of layer $i$ is the input of layer $i+1$, we can apply the chain rule to update the parameters of each layer simultenously. The following outlines the mathematical properties of this algorithm for each type of layer and the loss function from an algebraic perspective rather than the common (in my opinion, less intuitive), geometric perspective.

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

As mentioned above, the linear layer outputs a linear transformation of its input. Here we will look at the *batched* implementation, where the model processes a batch of $p$ samplpes in parallel. This is the most used approach, also known as mini-batch gradient descent. It is more computationally efficient and robust to noise within the samples since the learning steps are averaged between the samples.

Components:

1. Learnable weight matrix $\mathbf{W}\in\mathbb{R}^{n\times m}$
2. Bias vector $\mathbf{b}\in\mathbb{R}^{1\times m}$
3. Input matrix $\mathbf{X}\in\mathbb{R}^{p\times n}$
4. Output matrix $\mathbf{Y}\in\mathbb{R}^{p\times m}$

$$
\mathbf{W}= \begin{bmatrix} w_{11} & w_{12} & \cdots & w_{1m} \\ w_{21} & w_{22} & \cdots & w_{2m} \\ \vdots & \vdots & \ddots & \vdots \\ w_{n1} & w_{n2} & \cdots & w_{nm} \end{bmatrix}, \quad\mathbf{b}=\begin{bmatrix}b_1&b_2&\cdots&b_m\end{bmatrix}, \quad \mathbf{X}=\begin{bmatrix} x_{11} & x_{12} & \cdots & x_{1n} \\ x_{21} & x_{22} & \cdots & x_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ x_{b1} & x_{b2} & \cdots & x_{bn} \end{bmatrix}, \quad \mathbf{Y}=\begin{bmatrix} y_{11} & y_{12} & \cdots & w_{1m} \\ y_{21} & y_{22} & \cdots & y_{2m} \\ \vdots & \vdots & \ddots & \vdots \\ y_{b1} & y_{b2} & \cdots & y_{bm} \end{bmatrix}
$$

### Forward Pass

The output $\mathbf{Y}$ is calculated as follows:

$$
\mathbf{Y} = \mathbf{X}\mathbf{W}+\mathbf{b}
$$

Here the bias vector is added element-wise to each row of $\mathbf{XW}$. To be maximally specific, the output's elements are calculated as follows:

$$
y_{ij}=b_j+\sum_{k=1}^nx_{ik}w_{kj},\quad i=1,\ldots,p,\quad j=1,\ldots,m
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

### Backward Pass

The input is the derivative of the loss function $\mathcal{L}$ with respect to each component of the output $\mathbf{Y}$, i.e. $\frac{\partial\mathcal{L}}{\partial \mathbf{Y}}\in\mathbb{R}^{b\times m}$, which is defined as:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{Y}} = \begin{bmatrix} \frac{\partial \mathcal{L}}{\partial y_{11}} & \cdots & \frac{\partial \mathcal{L}}{\partial y_{1m}}\\\vdots&\ddots&\vdots\\\frac{\partial\mathcal{L}}{y_{b1}}&\cdots&\frac{\partial\mathcal{L}}{y_{bm}} \end{bmatrix}
$$

Using this, we need to find the derivative of the loss with respect to the weights, the bias, and the input. The first 2 are used to update the model's parameters, while the latter is the derivative of the loss with respect to the outputs of the previous layer, so it is directly passed to it and the same operations are repeated. This can be applied recursively to update all layers.

#### Derivative of the loss with respect to the weights

The derivative of the loss with respect to the weights, i.e. $\frac{\partial\mathcal{L}}{\partial\mathbf{W}}\in\mathbb{R}^{n\times m}$ is defined as:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{W}} = \begin{bmatrix} \frac{\partial \mathcal{L}}{\partial w_{11}} & \cdots & \frac{\partial \mathcal{L}}{\partial w_{1m}} \\ \vdots & \ddots & \vdots \\ \frac{\partial \mathcal{L}}{\partial w_{n1}} & \cdots & \frac{\partial \mathcal{L}}{\partial w_{nm}}\end{bmatrix}
$$

Here we can apply the chain rule:

$$
\frac{\partial\mathcal{L}}{\partial\mathbf{W}}=\frac{\partial\mathcal{L}}{\partial\mathbf{Y}}\frac{\partial\mathbf{Y}}{\partial\mathbf{W}}
$$

Since $\frac{\partial\mathcal{L}}{\partial\mathbf{Y}}$ is already known, we only need to find $\frac{\partial\mathbf{Y}}{\partial\mathbf{X}}$, i.e. the derivative of the output with respect to the weights. Normally the derivative of a matrix with respect to another matrix is not conventionally defined. In backpropagation, the gradients with respect to the weights are averaged over all samples in the input. Since the loss function already performs averaging, we can simply sum the gradients, which for the individual weights $(k,j)$ this is defined as follows:

$$
\frac{\partial\mathcal{L}}{\partial w_{kj}}=\sum_{i=1}^p\left(\begin{bmatrix}\frac{\partial\mathcal{L}}{\partial y_{i1}}&\cdots&\frac{\partial\mathcal{L}}{\partial y_{im}}\end{bmatrix}\begin{bmatrix}\frac{\partial y_{i1}}{\partial w_{kj}}\\\vdots\\\frac{\partial y_{im}}{\partial w_{kj}}\end{bmatrix}\right) = \sum_{i=1}^p\left(\frac{\partial\mathcal{L}}{\partial y_i}\frac{\partial y_i}{\partial w_{kj}}\right)
$$

where $y_i$ is the $i$-th row of $\mathbf{Y}$. Now we only need to find $\frac{\partial y_i}{\partial w_{kj}}$. Looking at each element, namely $\frac{\partial y_{il}}{\partial w_{kj}}$, we can immediately see the derivatve by looking at the forward pass. It is as follows:

$$
\frac{\partial y_{il}}{\partial w_{kj}}=\begin{cases}x_{ik} & \text{if } l=j \\ 0 & \text{otherwise}\end{cases}
$$

Now we can simply substitute back into the formula to get: 

$$
\frac{\partial\mathcal{L}}{\partial w_{kj}}=\sum_{i=1}^p\left(\begin{bmatrix}\frac{\partial\mathcal{L}}{\partial y_{i1}} & \cdots & \frac{\partial\mathcal{L}}{\partial y_{im}}\end{bmatrix}\begin{bmatrix}0\\\vdots\\ x_{ik}\\\vdots\\0\end{bmatrix}\right)=\sum_{i=1}^p\left(\frac{\partial\mathcal{L}}{\partial y_{ij}}x_{ik}\right)
$$

Intuitively, $\frac{\partial\mathcal{L}}{\partial w_{kj}}$ is equal to the dot product of $\mathbf{X}$'s $k$-th column and $\frac{\partial\mathcal{L}}{\partial\mathbf{Y}}$'s $j$-th. Now we can generalize for all weights as follows:

$$
\frac{\partial\mathcal{L}}{\partial\mathbf{W}}=\begin{bmatrix}\sum_{i=1}^p\frac{\partial\mathcal{L}}{\partial y_{i1}}x_{i1}&\cdots&\sum_{i=1}^p\frac{\partial\mathcal{L}}{\partial y_{im}}x_{i1}\\\vdots&\ddots&\vdots\\\sum_{i=1}^p\frac{\partial\mathcal{L}}{\partial y_{i1}}x_{im}&\cdots&\sum_{i=1}^p\frac{\partial\mathcal{L}}{\partial y_{im}}x_{im}\end{bmatrix}=\mathbf{X}^\top\frac{\partial\mathcal{L}}{\partial\mathbf{Y}}
$$

Now that we know $\frac{\partial \mathcal{L}}{\partial \mathbf{W}}$, we can update the weights by taking a small step in the opposite direction as follows:

$$
\mathbf{W}\gets\mathbf{W}-\alpha\frac{\partial \mathcal{L}}{\partial \mathbf{W}}
$$

#### Derivative of the loss with respect to the bias

The derivative of the loss with respect to the bias, i.e. $\frac{\partial\mathcal{L}}{\partial\mathbf{b}}\in\mathbb{R}^{1\times m}$ is defined as:

$$
\frac{\partial\mathcal{L}}{\partial\mathbf{b}}=\begin{bmatrix}\frac{\partial\mathcal{L}}{\partial b_1}&\cdots&\frac{\partial\mathcal{L}}{\partial b_1}\end{bmatrix}
$$

Again, applying the chain rule and summing over the input samples gives the following for each bias element $b_j$:

$$
\frac{\partial\mathcal{L}}{\partial b_j}=\sum_{i=1}^p\left(\begin{bmatrix}\frac{\partial\mathcal{L}}{\partial y_{i1}}&\cdots&\frac{\partial\mathcal{L}}{\partial y_{im}}\end{bmatrix}\begin{bmatrix}\frac{\partial y_{i1}}{\partial b_j}\\\vdots\\\frac{\partial y_{im}}{\partial b_j}\end{bmatrix}\right) = \sum_{i=1}^p\left(\frac{\partial\mathcal{L}}{\partial y_i}\frac{\partial y_i}{\partial b_j}\right)
$$

Again, we only need to find $\frac{\partial y_i}{\partial b_j}$. Looking at the individual elements $\frac{\partial y_{il}}{\partial b_j}$, the derivative according to the forward pass is as follows:

$$
\frac{\partial y_{il}}{\partial b_j}=\begin{cases}1 & \text{if } l=j\\0&\text{otherwise}\end{cases}
$$

Substituting into the formula we get:

$$
\frac{\partial\mathcal{L}}{\partial b_j}=\sum_{i=1}^p\left(\begin{bmatrix}\frac{\partial\mathcal{L}}{\partial y_{i1}} & \cdots & \frac{\partial\mathcal{L}}{\partial y_{im}}\end{bmatrix}\begin{bmatrix}0\\\vdots\\1\\\vdots\\0\end{bmatrix}\right)=\sum_{i=1}^p\frac{\partial\mathcal{L}}{\partial y_{ij}}
$$

Generalizing for all elements of $\mathbf{b}$, we get:

$$
\frac{\partial\mathcal{L}}{\partial\mathbf{b}}=\begin{bmatrix}\sum_{i=1}^p\frac{\partial\mathcal{L}}{\partial y_{i1}}&\cdots&\sum_{i=1}^p\frac{\partial\mathcal{L}}{\partial y_{im}}\end{bmatrix}
$$

Intuitively, this is just the sum of each column of $\frac{\partial\mathcal{L}}{\partial\mathbf{Y}}$, which is already known. Analogically to the weights, we can update the bias by taking a small step in the opposite direction of the gradient as follows:

$$
\mathbf{b}\gets\mathbf{b}-\alpha\frac{\partial\mathcal{L}}{\partial \mathbf{b}}
$$

#### Derivative of the loss with respect to the input

Finally, we need to find the derivative of the loss with respect to the input, namely, $\frac{\partial\mathcal{L}}{\partial\mathbf{X}}\in\mathbb{R}^{p\times n}$, which is defined as:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{X}} = \begin{bmatrix} \frac{\partial \mathcal{L}}{\partial x_{11}} & \cdots & \frac{\partial \mathcal{L}}{\partial x_{1n}}\\\vdots&\ddots&\vdots\\\frac{\partial\mathcal{L}}{x_{p1}}&\cdots&\frac{\partial\mathcal{L}}{x_{pn}} \end{bmatrix}
$$

We can again apply the chain rule to get:

$$
\frac{\partial\mathcal{L}}{\partial\mathbf{X}}=\frac{\partial\mathcal{L}}{\partial\mathbf{Y}}\frac{\partial\mathbf{Y}}{\partial\mathbf{X}}
$$

Since the derivative of a matrix with respect to another matrix is undefined, to make progress backpropagation uses the intuition that the $i$-th row of $\mathbf{Y}$ is entirely dependent on the whole weights matrix $\mathbf{W}$ and ONLY on the $i$-th row of $\mathbf{X}$. This means that we can find the derivatives of the loss with respect to each individual sample $i$ and simply stack them vertically. Formally, we can express this as follows:

$$
\frac{\partial\mathcal{L}}{\partial\mathbf{X}}=\begin{bmatrix}\frac{\partial\mathcal{L}}{\partial y_1}\frac{\partial\mathcal{y_1}}{\partial x_1}\\\vdots\\\frac{\partial\mathcal{L}}{\partial y_b}\frac{\partial\mathcal{y_b}}{\partial x_b}\end{bmatrix}
$$

Here $x_i$ and $y_i$ are the $i$-th rows of $\mathbf{X}$ and $\mathbf{Y}$ respectively. It can be seen that $\frac{\partial\mathcal{L}}{\partial y_i}$ is an $m$-dimensional vector and $\frac{\partial y_i}{\partial x_i}$ is an $m\times n$ matrix. Therefore, each element in the gradient is an $n$-dimensional vector, giving us the original dimensions of $\mathbf{X}$, namely, $p\times n$. Now all we need is to find $\frac{\partial y_i}{\partial x_i}$.

Looking at the individual elements $\frac{\partial y_{ij}}{\partial x_{ik}}$, we can immediately see from the forward pass that:

$$
\frac{\partial y_{ij}}{\partial x_{ik}}=w_{kj}
$$

Now generalizing for the entire matrix $\frac{\partial y_i}{\partial x_i}$ gives:

$$
\frac{\partial y_i}{\partial x_i}=\begin{bmatrix}w_{11}&\cdots&w_{n1}\\\vdots&\ddots&\vdots\\w_{1m}&\cdots&w_{nm}\end{bmatrix}=\mathbf{W}
^\top
$$

Now we can generalize to each element of $\frac{\partial\mathcal{L}}{\partial\mathbf{X}}$ as follows:

$$
\frac{\partial\mathcal{L}}{\partial\mathbf{X}}=\begin{bmatrix}\frac{\partial\mathcal{L}}{\partial y_1}\mathbf{W}^\top\\\vdots\\\frac{\partial\mathcal{L}}{\partial y_b}\mathbf{W}^\top\end{bmatrix} = \frac{\partial\mathcal{L}}{\partial\mathbf{Y}}\mathbf{W}^\top
$$

Now that we know $\frac{\partial \mathcal{L}}{\partial \mathbf{X}}$, we also know $\frac{\partial \mathcal{L}}{\partial \mathbf{Y}}$ of the previous layer since the current layer's input *is* the previous layer's output. That way we can continue training! In Python, the backward pass can be implemented as follows:

```python
    def backward(self, dEdy: np.ndarray) -> np.ndarray:
        """Update the parameters and propagate the error"""

        # Calculate the derivative of the error with respect to the weights 
        dEdW = self.x.T @ dEdy

        # The error with respect to the bias
        dEdb = dEdy.sum(axis=0)

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

1. Input matrix $\mathbf{X}\in\mathbb{R}^{p\times n}$
2. Output matrix $\mathbf{Y}\in\mathbb{R}^{p\times n}$
3. Non-linear activation function $f:\mathbb{R}\rightarrow \mathbb{R}$

### Forward pass

The output $\mathbf{Y}$ is calculated as $y_{ij}=f(x_{ij})$, i.e.

$$
\mathbf{Y} = \begin{bmatrix} f(x_{11}) & f(x_{12}) & \cdots & f(x_{1n}) \\ f(x_{21}) & f(x_{22}) & \cdots & f(x_{2n}) \\ \vdots & \vdots & \ddots & \vdots \\ f(x_{p1}) & f(x_{p2}) & \cdots & f(x_{pn})\end{bmatrix}
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

Since the activation layer has no learnable parameters, we only need to find the derivative of the loss $\mathcal{L}$ with respect to the input $\mathbf{X}$, i.e. $\frac{\partial \mathcal{L}}{\partial \mathbf{X}}\in\mathbb{R}^{p\times n}$, and pass it to the previous layer. It is defined as

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{X}} = \begin{bmatrix} \frac{\partial \mathcal{L}}{\partial x_{11}} & \cdots & \frac{\partial \mathcal{L}}{\partial x_{1n}} \\ \vdots & \ddots & \vdots \\ \frac{\partial \mathcal{L}}{\partial x_{p1}} & \cdots & \frac{\partial \mathcal{L}}{\partial x_{pn}} \end{bmatrix}
$$

Applying the chain rule to each component gives

$$
\frac{\partial \mathcal{L}}{\partial x_{ij}}=\frac{\partial \mathcal{L}}{\partial \mathbf{Y}}\frac{\partial \mathbf{Y}}{\partial x_{ij}}
$$

Similarly to the backward pass of the linear layer, it can be observed that every output element $y_{ij}$ depends ONLY on $x_{ij}$. Therefore, when calculating $\frac{\partial\mathcal{L}}{\partial x_{ij}}$, we can consider only $y_{ij}$ instead of the entire matrix $\mathbf{Y}$. Therefore, we can express the gradient as follows:

$$
\frac{\partial\mathcal{L}}{\partial x_{ij}}=\frac{\partial \mathcal{L}}{\partial y_{ij}}\frac{\partial y_{ij}}{\partial x_{ij}}
$$

From the formula for the forward pass it can be seen that:

$$
\frac{\partial y_{ij}}{\partial x_{ij}}=f^\prime(x_{ij})
$$

Generalizing for all inputs we get:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{X}} = \begin{bmatrix} \frac{\partial \mathcal{L}}{\partial y_{11}}f^\prime(x_{11}) & \cdots & \frac{\partial \mathcal{L}}{\partial y_{1n}}f^\prime(x_{1n}) \\ \vdots & \ddots & \vdots \\ \frac{\partial \mathcal{L}}{\partial y_{p1}}f^\prime(x_{p1}) & \cdots & \frac{\partial \mathcal{L}}{\partial y_{pn}}f^\prime(x_{pn}) \end{bmatrix}=\frac{\partial\mathcal{L}}{\partial\mathbf{Y}}\odot f^\prime(\mathbf{X})
$$

Now, $\frac{\partial \mathcal{L}}{\partial \mathbf{X}}$ is passed to the previous layer, similarly to the linear layer.

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

1. Predictions matrix $\hat{\mathbf{Y}}\in\mathbb{R}^{p\times n}$
2. Ground truth matrix $\mathbf{Y}\in\mathbb{R}^{p\times n}$

This project focuses on the Mean Squared Error loss, $\text{MSE}:\mathbb{R}^{p\times n}\times\mathbb{R}^{p\times n}\rightarrow \mathbb{R}$, defined as

$$
\text{MSE}(\mathbf{Y},\hat{\mathbf{Y}})=\frac{1}{pn}\sum_{i=1}^p\sum_{j=1}^n(y_{ij}-\hat{y}_{ij})^2
$$

We need to find the derivative of the loss $\mathcal{L}$, computed by $\text{MSE}$, with respect to the predictions $\hat{\mathbf{Y}}$, defined as

$$
\frac{\partial \mathcal{L}}{\partial \hat{\mathbf{Y}}} = \begin{bmatrix} \frac{\partial \mathcal{L}}{\partial \hat{y}_{11}} & \cdots & \frac{\partial \mathcal{L}}{\partial \hat{y}_{1n}}\\\vdots&\ddots&\vdots\\\frac{\partial \mathcal{L}}{\partial \hat{y}_{p1}} & \cdots & \frac{\partial \mathcal{L}}{\partial \hat{y}_{pn}} \end{bmatrix}
$$

Now we can look at each component

$$
\frac{\partial \mathcal{L}}{\partial \hat{y}_{kl}}=\frac{\partial}{\partial \hat{y}_{kl}}\left(\frac{1}{pn}\sum_{i=1}^p\sum_{j=1}^n(y_{ij}-\hat{y}_{ij})^2\right) = \frac{1}{pn}\sum_{i=1}^p\sum_{j=1}^n\frac{\partial}{\partial{\hat{y}_{kl}}}(y_{ij}-\hat{y}_{ij})^2
$$

Above I extracted the common factor $\frac{1}{pn}$ and pushed $\frac{\partial}{\partial \hat{y}_{kl}}$ into the sums (which is fine due to the linearity of the differentiation operation). Now we know that $\frac{\partial}{\partial \hat{y}_{kl}}(y_{ij}-\hat{y}_{ij})^2\neq 0$ iff $i=k$ and $j=l$. Therefore, there is only 1 non-zero term in the sum, meaning that we can simplify as follows:

$$
\frac{\partial \mathcal{L}}{\partial \hat{y}_{kl}}=\frac{1}{pn}\frac{\partial}{\partial \hat{y}_{kl}}(y_{kl}-\hat{y}_{kl})^2=\frac{2}{pn}(\hat{y}_{kl}-y_{kl})
$$

Substituting back to $\frac{\partial \mathcal{L}}{\partial \hat{\mathbf{Y}}}$, we get

$$
\frac{\partial \mathcal{L}}{\partial \hat{\mathbf{Y}}} = \begin{bmatrix} \frac{2}{pn}(\hat{y}_{11}-y_{11}) & \cdots & \frac{2}{n}(\hat{y}_{1n}-y_{1n})\\\vdots&\ddots&\vdots\\\frac{2}{pn}(\hat{y}_{p1}-y_{p1}) & \cdots & \frac{2}{n}(\hat{y}_{pn}-y_{pn}) \end{bmatrix} = \frac{2}{pn}(\hat{\mathbf{Y}}-\mathbf{Y})
$$

The result is $\frac{\partial \mathcal{L}}{\partial \mathbf{Y}}$ of the final layer.

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
        epochs: int = 1_000, 
        store_all: bool = False
    ) -> Union[None, Tuple[np.ndarray, np.ndarray]]:
        """Train the NN"""

        for epoch in tqdm(range(epochs)):

            # Do the forward pass
            y_pred = self.predict(X)
            
            # Calculate the loss
            loss = self.loss(y, y_pred)
            
            # Calculate the gradient of the loss with respect to the outputs
            error = self.loss_prime(y, y_pred)

            # Sequentially update the layers with backpropagation
            for layer in reversed(self.layers): error = layer.backward(error)
```

Here `X` is a matrix where each row is an input sample and each column is a feature, and `y` is a vector representing the ground truth labels for each sample. Also, `epochs` is the number of times the model sees the entire dataset. It can be seen that the model processes and trains on each input independently. In practice, it is useful to use batches, where the model can process multiple outputs at once using efficient vectorized operations. Then, the propagated gradient is averaged over the entire batch. For simplicity (especially the mathy part), this project implements only the single-input gradient descent.

## Interaction

Try out the neural network itself with the following interaction!
