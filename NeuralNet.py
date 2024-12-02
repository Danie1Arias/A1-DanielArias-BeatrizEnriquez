import numpy as np

class NeuralNet:
  def __init__(self, layers, epochs, learning_rate, momentum, fact):
    num_layers = len(layers)
    num_units = layers.copy()

    self.L = num_layers
    self.n = num_units
    self.epochs = epochs
    self.eta = learning_rate
    self.alpha = momentum
    self.fact = self._get_fact(fact)
    self.d_fact = self._get_d_fact(fact)
    self.xi = [np.zeros(layer_units) for layer_units in num_units]
    self.h = [np.zeros(layer_units) for layer_units in num_units]
    self.w = [None] + [np.zeros((num_units[i], num_units[i - 1])) for i in range(1, num_layers)]
    self.theta = [np.zeros(layer_units) for layer_units in num_units]
    self.delta = [np.zeros(layer_units) for layer_units in num_units]
    self.d_w = [None] + [np.zeros((num_units[i], num_units[i - 1])) for i in range(1, num_layers)]
    self.d_theta = [np.zeros(layer_units) for layer_units in num_units]
    self.d_w_prev = [None] + [np.zeros((num_units[i], num_units[i - 1])) for i in range(1, num_layers)]
    self.d_theta_prev = [np.zeros(layer_units) for layer_units in num_units]
    self.training_error = []
    self.validation_error = []


  def _get_fact(self, fact):
    if fact == "sigmoid":
        return lambda x: 1 / (1 + np.exp(-x))
    elif fact == "relu":
        return lambda x: np.maximum(0, x)
    elif fact == "linear":
        return lambda x: x
    else:
        return lambda x: np.tanh(x)
    
  
  def _get_d_fact(self, fact):
    if fact == "sigmoid":
        return lambda x: x * (1 - x)
    elif fact == "relu":
        return lambda x: (x > 0).astype(float)
    elif fact == "linear":
        return lambda x: np.ones_like(x)
    else:
        return lambda x: 1 - x ** 2
      

  def fit(self, X_train, y_train, X_val, y_val):
    rows, cols = X_train.shape

    for epoch in range(self.epochs):
        epoch_errors = []

        for row in range(rows):
            self._feed_forward(X_train[row])
            self._backpropagate(y_train[row])
            self._update_weights_and_thresholds()

        train_errors = self._calculate_total_error(X_train, y_train)
        epoch_errors.append(np.mean(train_errors))

        val_errors = None
        if X_val.shape[0] > 0:
           val_errors = self._calculate_total_error(X_val, y_val)
           epoch_errors.append(np.mean(val_errors))
        else:
           epoch_errors.append(0)
           val_errors = []

        self.training_error.append(np.mean(train_errors))
        self.validation_error.append(np.mean(val_errors) if val_errors is not None else 0)

        max_len = min(len(self.training_error), len(self.validation_error), len(epoch_errors))
        epoch_errors += [0] * (max_len - len(epoch_errors))
        self.validation_error += [0] * (max_len - len(self.validation_error))


  def _feed_forward(self, X):
    self.xi[0] = X

    for l in range(1, self.L):
        self.h[l] = np.dot(self.w[l], self.xi[l-1]) - self.theta[l]
        self.xi[l] = self.fact(self.h[l])


  def _backpropagate(self, y):
    self.delta[self.L - 1] = self.d_fact(self.xi[self.L - 1]) * (self.xi[self.L - 1] - y)

    for l in range(self.L - 2, 0, -1):
      self.delta[l] = self.d_fact(self.xi[l]) * np.dot(self.w[l + 1].T, self.delta[l + 1])


  def _update_weights_and_thresholds(self):
    for l in range(1, self.L):
        self.d_w[l] = -self.eta * np.outer(self.delta[l], self.xi[l - 1]) + self.alpha * self.d_w_prev[l]
        self.d_theta[l] = self.eta * self.delta[l] + self.alpha * self.d_theta_prev[l]
        self.w[l] += self.d_w[l]
        self.theta[l] += self.d_theta[l]
        self.d_w_prev[l] = self.d_w[l]
        self.d_theta_prev[l] = self.d_theta[l]

  
  def _calculate_total_error(self, X, y):
    errors = []
    for i in range(X.shape[0]):
        self._feed_forward(X[i])
        error = 0.5 * np.sum((self.xi[self.L - 1] - y[i]) ** 2)
        errors.append(error)
    return errors


  def predict(self, X):
    predictions = []
    for sample in X:
        self._feed_forward(sample)
        prediction = self.xi[self.L - 1].copy()
        prediction[np.isnan(prediction)] = 0
        predictions.append(prediction)
    return np.array(predictions)


  def loss_epochs(self):
        return np.column_stack((self.training_error, self.validation_error))
    