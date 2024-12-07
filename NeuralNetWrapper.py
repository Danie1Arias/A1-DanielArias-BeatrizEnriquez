from sklearn.base import BaseEstimator
import NeuralNet as nn
import numpy as np

class NeuralNetWrapper(BaseEstimator):
    def __init__(self, layers, epochs=100, learning_rate=0.01, momentum=0.9, fact="sigmoid", n_splits=2):
        self.layers = layers
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.fact = fact
        self.n_splits = n_splits
        self.model = nn.NeuralNet(layers, epochs, learning_rate, momentum, fact, n_splits)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        predictions = self.predict(X)
        mse = np.mean((predictions - y) ** 2)
        return -mse

    def get_params(self, deep=True):
        return {
            "layers": self.layers,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "momentum": self.momentum,
            "fact": self.fact,
            "n_splits": self.n_splits,
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)

        self.model = nn.NeuralNet(
            self.layers,
            self.epochs,
            self.learning_rate,
            self.momentum,
            self.fact,
            self.n_splits,
        )
        
        return self
