import numpy as np
from sklearn.model_selection import KFold

"""
It's important to initialize the hidden layers' connection weights randomly, or else training will fail.
For example, if you initialize weights and biases to zero, then all neurons in a given layer will be
perfectly identical, and thus, backpropagation will affect them in exactly the same way, so they will
remian identical.
"""

class NeuralNet:
  def __init__(self, layers, epochs, learning_rate, momentum, fact, n_splits=2):
      num_layers = len(layers)
      num_units = layers.copy()

      self.L = num_layers
      self.n = num_units
      self.epochs = epochs
      self.eta = learning_rate
      self.alpha = momentum
      self.fact = self._get_fact(fact)
      self.d_fact = self._get_d_fact(fact)
      self.n_splits = 2 if n_splits < 2 else int(np.ceil(n_splits))
  
      # Standard Normal Distribution + He initialization (Optimal for ReLU activation functions)
      self.w = [None] + [np.random.randn(num_units[i], num_units[i - 1]) * np.sqrt(2 / num_units[i - 1]) for i in range(1, num_layers)]  # Pesos con inicialización He
      
      self.xi = [np.zeros(layer_units) for layer_units in num_units]
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
      

  def fit(self, X, y):
    kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)

    for epoch in range(self.epochs):
      fold_train_errors = []
      fold_val_errors = []

      for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Number of patterns in the training set
        patterns = X_train.shape[0]

        # Train on the training set
        for pattern in range(patterns):
          self._feed_forward(X_train[pattern])
          self._backpropagate(y_train[pattern])
          self._update_weights_and_thresholds()

        # Calculate errors per fold
        train_errors = self._calculate_total_error(X_train, y_train)
        fold_train_errors.append(np.mean(train_errors))

        val_errors = self._calculate_total_error(X_test, y_test)
        fold_val_errors.append(np.mean(val_errors))

      # Average errors between folds
      epoch_train_error = np.mean(fold_train_errors)
      epoch_val_error = np.mean(fold_val_errors)

      # Store errors for plotting
      self.training_error.append(epoch_train_error)
      self.validation_error.append(epoch_val_error)



  def _feed_forward(self, X):
    # Assign input values to the first layer
    self.xi[0] = X

    # Hidden layers (With activation functions)
    for l in range(1, self.L - 1):
        self.xi[l] = self.fact(np.dot(self.w[l], self.xi[l - 1]) - self.theta[l])

    # Output layer (Without activation functions)
    self.xi[self.L - 1] = np.dot(self.w[self.L - 1], self.xi[self.L - 2]) - self.theta[self.L - 1]


  def _backpropagate(self, y):
    # Calculate the error for the output layer using the squared error derivative formula
    self.delta[self.L - 1] = self.d_fact(self.xi[self.L - 1]) * (self.xi[self.L - 1] - y)

    # Backpropagation from the penultimate layer to the first
    for l in range(self.L - 2, 0, -1):
      self.delta[l] = self.d_fact(self.xi[l]) * np.dot(self.w[l + 1].T, self.delta[l + 1])


  def _update_weights_and_thresholds(self):
    for l in range(1, self.L):
        # Update weights with momentum
        # Calculate the gradient using the derivative rule of the error with respect the weights
        self.d_w[l] = -self.eta * np.outer(self.delta[l], self.xi[l - 1]) + self.alpha * self.d_w_prev[l]
        self.w[l] += self.d_w[l]

        # Update thresholds with momentum
        self.d_theta[l] = self.eta * self.delta[l] + self.alpha * self.d_theta_prev[l]
        self.theta[l] += self.d_theta[l]

        # Save previous values ​​for momentum
        self.d_w_prev[l] = self.d_w[l]
        self.d_theta_prev[l] = self.d_theta[l]

  
  def _calculate_total_error(self, X, y):
    errors = []

    for i in range(X.shape[0]):
        # Forward propagation for each pattern
        self._feed_forward(X[i])

        # Calculate the mean square error (MSE) for the pattern
        error = 0.5 * np.sum((self.xi[self.L - 1] - y[i]) ** 2)
        errors.append(error)

    # Calculate and return the average error
    return np.mean(errors)


  def predict(self, X):
      predictions = []
      
      for sample in X:
          self._feed_forward(sample)  
          prediction = self.xi[self.L - 1]
          predictions.append(prediction)
      
      return np.array(predictions)
  

  def loss_epochs(self):
    # Verify that error lists are the same size
    min_len = min(len(self.training_error), len(self.validation_error))

    # Trim lists to minimum length if necessary
    training_error_trimmed = self.training_error[:min_len]
    validation_error_trimmed = self.validation_error[:min_len]

    # Return training and validation errors as two separate lists
    return np.array(training_error_trimmed), np.array(validation_error_trimmed)
