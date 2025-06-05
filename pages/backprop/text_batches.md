# Backpropagation With Batches

In practice, neural networks often work with batches of inputs. That is, they can process a stack of inputs in parallel both in the forward and backward pass. This is both more computationally efficient and robust to noise in the samples since the update steps are averaged over the batch. Moreover, we need to make only one tiny change to the code from before. Let's look at the maths first!

Firstly, we should replace the input vectors $\mathbf{x}$ and output vectors $\mathbf{y}$ with $p\times n$ and $p\times m$ matrices $\mathbf{X}$ and $\mathbf{Y}$ respectively, there $p$ is the number of samples in the batch. They are defined as follows:

$$
\mathbf{X}=\begin{bmatrix} x_{11} & x_{12} & \cdots & x_{1n} \\ x_{21} & x_{22} & \cdots & x_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ x_{p1} & x_{p2} & \cdots & x_{pn} \end{bmatrix}, \quad \mathbf{Y}=\begin{bmatrix} y_{11} & y_{12} & \cdots & w_{1m} \\ y_{21} & y_{22} & \cdots & y_{2m} \\ \vdots & \vdots & \ddots & \vdots \\ y_{p1} & y_{p2} & \cdots & y_{pm} \end{bmatrix}
$$

Now we can look again at each layer separately.

## Linear Layer

The linear layer is largely the same as before

### Forward Pass

The output $\mathbf{Y}$ is calculated as follows:

$$
\mathbf{Y} = \mathbf{X}\mathbf{W}+\mathbf{b}
$$

Here the bias vector is added element-wise to each row of $\mathbf{XW}$. To be maximally specific, the output's elements are calculated as follows:

$$
y_{ij}=b_j+\sum_{k=1}^nx_{ik}w_{kj},\quad i=1,\ldots,p,\quad j=1,\ldots,m
$$

Matrix multiplication can be directly used here as the output for sample $i$ is simply the vector-matrix product between the $i$-th input sample and the weight matrix.

### Backward Pass

Now the derivative of the loss function $\mathcal{L}$ with respect to each component of the output $\mathbf{Y}$, is a matrix, i.e. $\frac{\partial\mathcal{L}}{\partial \mathbf{Y}}\in\mathbb{R}^{b\times m}$, which is defined as:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{Y}} = \begin{bmatrix} \frac{\partial \mathcal{L}}{\partial y_{11}} & \cdots & \frac{\partial \mathcal{L}}{\partial y_{1m}}\\\vdots&\ddots&\vdots\\\frac{\partial\mathcal{L}}{y_{b1}}&\cdots&\frac{\partial\mathcal{L}}{y_{bm}} \end{bmatrix}
$$

#### Derivative of the loss with respect to the weights

The derivative of the loss with respect to the weights is defined the same as before. To find it, we start by applying the chain rule:

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

We can see that the expression is exactly the same as before, only the input vector $\mathbf{x}$ has become the input matrix $\mathbf{X}$. No change to the code is needed to implement batches.

#### Derivative of the loss with respect to the bias

The derivative of the loss with respect to the bias is defined the same as before. To find it, we start by applying the chain rule and summing over the input samples, similarly to the weights. This gives the following for each bias element $b_j$:

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

Intuitively, this is just the sum of each column of $\frac{\partial\mathcal{L}}{\partial\mathbf{Y}}$, which is already known. The summation is the only necessary change to the code:

```python
        # The error with respect to the bias
        dEdb = dEdy.sum(axis=0)
```

#### Derivative of the loss with respect to the input

Finally, we need to find the derivative of the loss with respect to the input, namely, $\frac{\partial\mathcal{L}}{\partial\mathbf{X}}\in\mathbb{R}^{p\times n}$, which is now defined as:

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

Amazing! The expression is the same as before, no change to the code is needed.

## Activation Layer

The activation layer also remains largely the same as above.

### Forward pass

The output $\mathbf{Y}$ is calculated as $y_{ij}=f(x_{ij})$, i.e.

$$
\mathbf{Y} = \begin{bmatrix} f(x_{11}) & f(x_{12}) & \cdots & f(x_{1n}) \\ f(x_{21}) & f(x_{22}) & \cdots & f(x_{2n}) \\ \vdots & \vdots & \ddots & \vdots \\ f(x_{p1}) & f(x_{p2}) & \cdots & f(x_{pn})\end{bmatrix}
$$

Similarly to before, the output is defined by applying the unary activation function to each element in the input. No change in the code is required!

### Backward pass

Again, we only need to find the derivative of the loss $\mathcal{L}$ with respect to the input $\mathbf{X}$, i.e. $\frac{\partial \mathcal{L}}{\partial \mathbf{X}}\in\mathbb{R}^{p\times n}$, and pass it to the previous layer. It is defined as:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{X}} = \begin{bmatrix} \frac{\partial \mathcal{L}}{\partial x_{11}} & \cdots & \frac{\partial \mathcal{L}}{\partial x_{1n}} \\ \vdots & \ddots & \vdots \\ \frac{\partial \mathcal{L}}{\partial x_{p1}} & \cdots & \frac{\partial \mathcal{L}}{\partial x_{pn}} \end{bmatrix}
$$

We can apply the chain rule but only consider $y_{ij}$ for each input $x_{ij}$. That is because $y_{ij}$ is entirely dependent only on $x_{ij}$ and $x_{ij}$ also does not influence any other outputs. Therefore, for each element, the chain rule is as follows:

$$
\frac{\partial \mathcal{L}}{\partial x_{ij}}=\frac{\partial \mathcal{L}}{\partial y_{ij}}\frac{\partial y_{ij}}{\partial x_{ij}}
$$

From the formula for the forward pass it can be seen that:

$$
\frac{\partial y_{ij}}{\partial x_{ij}}=f^\prime(x_{ij})
$$

Generalizing for all inputs we get:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{X}} = \begin{bmatrix} \frac{\partial \mathcal{L}}{\partial y_{11}}f^\prime(x_{11}) & \cdots & \frac{\partial \mathcal{L}}{\partial y_{1n}}f^\prime(x_{1n}) \\ \vdots & \ddots & \vdots \\ \frac{\partial \mathcal{L}}{\partial y_{p1}}f^\prime(x_{p1}) & \cdots & \frac{\partial \mathcal{L}}{\partial y_{pn}}f^\prime(x_{pn}) \end{bmatrix}=\frac{\partial\mathcal{L}}{\partial\mathbf{Y}}\odot f^\prime(\mathbf{X})
$$

Same expression as before! No change to the code is needed.

## Loss Function

The loss function can also be easily extended to matrices. It will now look as follows:

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

This is essentially the same expression as before! No change to the code needed.

## Training loop

With batches we can simplify our training loop as follows:

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

Now we can directly pass the entire dataset through the neural network and perform one update step by learning from all inputs at the same time!
