import numpy as np
from tqdm.auto import tqdm
import pickle

np.random.seed(0)

class BSpline:
    
    def __init__(self, p: int = 3, n: int = 7) -> None:
        """
        B-Spline
        
        Inputs:
        - p (int): The degree of each B-spline basis element
        - n (int): The number of basis elements
        """
        
        knots = n+p+1
        grid_size = n-p
        step = 2 / grid_size
        self.t = np.linspace(-1 - p * step, 1 + p * step, knots)
        self.p = p
        self.n = n

    def _rec_call(self, x: np.ndarray, p: int, i: int, t: np.ndarray) -> np.ndarray:
        if p == 0: return ((t[i] <= x) & (x < t[i+1])).astype(float)
        return (
            (x - t[i]) / (t[i+p] - t[i]) * self._rec_call(x, p-1, i, t)
            +
            (t[i+p+1] - x) / (t[i+p+1] - t[i+1]) * self._rec_call(x, p-1, i+1, t)
        )

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Recursively evaluate the B-Spline to get the contribution of the basis elements at x

        Inputs:
        - x (numpy.ndarray): An array of shape [m], the input

        Returns:
        - numpy.ndarray: Shape [n, m], element ij is the contribution of element i on point j in x
        """
        
        components = [self._rec_call(x, self.p, i, self.t) for i in range(self.n)]
        return np.stack(components)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Recursively evaluate the derivative of the B-Spline

        Inputs:
        - x (numpy.ndarray): An array of shape [m], the input

        Returns:
        - numpy.ndarray: Shape [n, m], element ij is the contribution of element i on point j in x
        """

        components = [
            (
                self._rec_call(x, self.p-1, i, self.t) / (self.t[i+self.p] - self.t[i])
                -
                self._rec_call(x, self.p-1, i+1, self.t) / (self.t[i+self.p+1] - self.t[i+1])
            )
            for i in range(self.n)
        ]
        return self.p * np.stack(components)

class Layer:

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
        
    def backward(self, dEdY: np.ndarray) -> np.ndarray:
        dEdW = np.einsum('j,ki->ijk', dEdY, self.B)
        self.W -= self.learning_rate * dEdW
        B_prime = self.spline.derivative(self.x)
        dEdX = np.einsum('j,ijk,ki->i', dEdY, self.W, B_prime)
        return dEdX

class Tanh:
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        return np.tanh(x)
    def backward(self, dEdY: np.ndarray) -> np.ndarray:
        dEdX = dEdY * (1 - np.tanh(self.x) ** 2)
        return dEdX

class MSE:    
    def forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return ((y_true - y_pred) ** 2).mean()

    def backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return 2 * (y_pred - y_true) / y_true.size

class NN:
    
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
            
if __name__ == '__main__':
    
    import matplotlib.pyplot as plt
    np.random.seed(0)
    nn = NN()
    print(sum(layer.W.size for layer in nn.layers if type(layer) == Layer))
    x_train = np.linspace(-1, 1, 30).reshape(-1, 1)
    y_train = .5*np.sin(4*x_train) * np.exp(-(x_train+1))
    nn.fit(x_train, y_train, epochs=100)
    with open('assets/kan/nn.pkl', 'wb') as f:
        pickle.dump(
            obj=[
                (layer.W, layer.spline.n, layer.spline.p)
                for layer in nn.layers
                if type(layer) == Layer
            ],
            file=f
        )
        # pickle.dump(nn, f)
    y_pred = [nn.predict(x) for x in x_train]
    np.save('assets/kan/y_pred', y_pred)
    np.save('assets/kan/y_train', y_train)
    # plt.plot(x_train.flatten(), y_train.flatten(), label='true')
    # plt.plot(x_train.flatten(), np.array(y_pred).flatten(), label='pred', linewidth=2)
    # plt.legend()
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.title('KAN learning evaluation\n' + fr'$y=\frac{{1}}{{2}}\sin(4x)\exp(-x-1)$' + '\nMSE: ' + f'{MSE().forward(y_train, y_pred):.4f}')
    # plt.tight_layout()
    # plt.savefig('training.svg')
    # plt.show()
    