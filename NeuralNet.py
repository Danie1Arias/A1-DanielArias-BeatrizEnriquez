import numpy as np
import random

class NeuralNet:
  def __init__(self, layers, epochs, learning_rate, momentum, fact):
    self.L = len(layers)
    self.n = layers.copy()
    self.n_epochs = epochs
    self.learning_rate = learning_rate
    self.momentum = momentum
    self.fact, self.d_fact = self._get_activation_function(fact)
    self.d_w = []
    self.d_w_prev = []
    self.d_theta = []
    self.theta_prev = []
    self.delta = []
    
    # Parameters to review
    self.train_loss = [] 
    self.val_loss = [] 

    self.xi = []
    for lay in range(self.L):
      self.xi.append(np.random.rand(layers[lay]))

    self.w = []
    self.theta = []
    for lay in range(1, self.L):
      self.w.append(np.random.rand(layers[lay], layers[lay - 1]))
      self.theta.append(np.random.rand(layers[lay], 1))

    self.v_w = [np.zeros_like(w) for w in self.w]
    self.v_theta = [np.zeros_like(t) for t in self.theta]


  def _get_activation_function(self, activation):
    if activation == "sigmoid":
        return lambda x: 1 / (1 + np.exp(-x)), lambda x: x * (1 - x)
    elif activation == "relu":
        return lambda x: np.maximum(0, x), lambda x: (x > 0).astype(float)
    elif activation == "linear":
        return lambda x: x, lambda x: np.ones_like(x)
    else:
        return lambda x: np.tanh(x), lambda x: 1 - x ** 2
      

  def fit(self, X, y):
    rows, cols = X.shape

    for epoch in range(self.n_epochs):
      for row in range(rows):

        a = self._forward(X[row])
        
        train_loss = np.mean((a[-1] - y.T) ** 2)
        self.train_loss.append(train_loss)

        # Validation
        val_a = self._forward(X)
        val_loss = np.mean((val_a[-1] - y.T) ** 2)
        self.val_loss.append(val_loss)

        # Back propagation
        dw, db = self._backward(a, y)

        # Update weights by using momentum
        for l in range(len(self.w)):
            v_w[l] = self.momentum * v_w[l] - self.lr * dw[l]
            v_theta[l] = self.momentum * v_theta[l] - self.lr * db[l]
            self.w[l] += v_w[l]
            self.b[l] += v_b[l]


  def _forward(self, X):
    a = [X.T]

    for l in range(self.L - 1):
        z = self.w[l] @ a[l] + self.b[l]
        a.append(self.fact(z))
    return a


  def _backward(self, a, y):
    m = y.shape[0]
    y = y.reshape(-1, 1).T
    dz = a[-1] - y
    dw = [(dz @ a[-2].T) / m]
    db = [np.sum(dz, axis=1, keepdims=True) / m]

    for l in range(self.L - 2, 0, -1):
        dz = (self.w[l].T @ dz) * self.d_fact(a[l])
        dw.insert(0, (dz @ a[l-1].T) / m)
        db.insert(0, np.sum(dz, axis=1, keepdims=True) / m)

    return dw, db
  

  def predict(self, X):
    a = self._forward(X)
    return a[-1].T
  

  def loss_epochs(self):
    return np.array(self.train_loss), np.array(self.val_loss)
  