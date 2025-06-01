import numpy as np
from typing import Callable, List, Union, Tuple
from tqdm.auto import tqdm
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=.001, help='Learning rate of the Neural Network')
    parser.add_argument('--activation', type=str, choices=['relu', 'tanh'], default='relu', help='Activation function of the layers')
    parser.add_argument('--n_layers', type=int, default=1, help='Number of hidden layers')
    parser.add_argument('--size', type=int, default=16, help='Number of neurons in each hidden layer')
    parser.add_argument('--n_samples', type=int, default=100, help='Number of samples on which to train the network')
    parser.add_argument('--n_epochs', type=int, default=1_000, help='Number of training epochs')
    parser.add_argument('--target_function', type=str, choices=['exp', 'log1p', 'sqrt'], default='exp', help='Function that the network approximates')
    return parser.parse_args()

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
    
    def backward(self, dEdy: np.ndarray) -> np.ndarray:
        """Update the parameters and propagate the error"""

        # Calculate the derivative of the error with respect to the weights 
        dEdW = self.x.T @ dEdy

        # The error with respect to the bias
        dEdb = dEdy.mean(axis=0)

        # The error with respect to the input
        dEdx = dEdy @ self.W.T

        # Update the parameters using gradient descent
        self.W -= self.lr * dEdW
        self.b -= self.lr * dEdb

        # Propagate the error
        return dEdx
    
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

    def backward(self, dEdy: np.ndarray) -> np.ndarray:
        """Propagate the error"""

        dEdx = dEdy * self.f_prime(self.x)
        return dEdx
    
def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)

def tanh_prime(x: np.ndarray) -> np.ndarray:
    return 1 - np.tanh(x) ** 2

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)

def relu_prime(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(float)

def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return ((y_true - y_pred) ** 2).mean()

def mse_prime(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return 2 * (y_pred - y_true) / y_true.size

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
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Given a sample x_i, predict the final output of the networks"""
        
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def fit(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        epochs: int = 1_000, 
        store_all: bool = False
    ) -> Union[None, Tuple[np.ndarray, np.ndarray]]:
        """Train the NN"""

        if store_all: 
            initial_preds = self.predict(X)
            predictions = [initial_preds]
            losses = [self.loss(y, initial_preds)]

        for epoch in tqdm(range(epochs)):

            # Do the forward pass
            y_pred = self.predict(X)
            
            # Calculate the loss
            loss = self.loss(y, y_pred)
            print(loss)
            # Calculate the gradient of the loss with respect to the outputs
            error = self.loss_prime(y, y_pred)

            # Sequentially update the layers with backpropagation
            for layer in reversed(self.layers): error = layer.backward(error)

            if store_all: predictions.append(y_pred)
            if store_all: losses.append(loss)

        if store_all:
            return np.array(losses), np.stack(predictions)

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    np.random.seed(1)
    args = parse_args()

    target_f = {'exp': np.exp, 'log1p': np.log1p, 'sqrt': np.sqrt}[args.target_function]
    nn = NN(activation=args.activation, n_layers=args.n_layers, size=args.size, lr=args.lr)
    x_train = np.linspace(0, 1, args.n_samples)[:, np.newaxis]
    # y_train = target_f(x_train)
    y_train = np.sin(4 * np.pi * x_train)
    nn.fit(x_train, y_train, epochs=args.n_epochs, store_all=True)
    
    preds = nn.predict(x_train)
    plt.plot(x_train.squeeze(), preds.squeeze(), label='prediction', linewidth=2, alpha=.7)
    plt.plot(x_train.squeeze(), y_train.squeeze(), label='ground truth', linewidth=2, alpha=.7)
    plt.xlabel('$x$')
    plt.ylabel('$e^x$')
    plt.title(f'Neural network fit to the $e^x$ function\nMSE: {mse(y_train, preds):.4f}')
    plt.legend()
    # plt.savefig('plot.svg')
    plt.show()
