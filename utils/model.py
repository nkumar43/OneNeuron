class Perceptron:
  def __init__(self, eta, epochs):
    self.weights = np.random.randn(3) * 1e-4      # Small Weight initialization
    print(f'Initial weights before training : \n {self.weights}')
    self.eta = eta
    self.epochs = epochs

  def activationFunction(self, inputs, weights):
    z = np.dot(inputs, weights)                   # z = W * X
    return np.where(z > 0, 1, 0)                  # (Condition, if True, Else)

  def fit(self, X, Y):
    self.X = X
    self.Y = Y

    X_with_bias = np.c_[self.X, -np.ones((len(self.X), 1))]
    print(f'X with bias : \n {X_with_bias}')

    for epoch in range(self.epochs):
      print('--'*10)
      print(f'For epoch : {epoch}')
      print('--'*10)

      y_hat = self.activationFunction(X_with_bias, self.weights) # Forward Propagation
      print(f'Predicted value after forward pass : \n {y_hat}')
      self.error = self.Y - y_hat
      print(f'Error : \n {self.error}')
      self.weights = self.weights + self.eta * np.dot(X_with_bias.T, self.error)   # Backward propagation weight update
      print(f'Updated weight after epoch : \n {epoch} / {self.epochs} : \n {self.weights}')
      print('####'*10)


  def predict(self, X):
    X_with_bias = np.c_[X, -np.ones((len(X), 1))]
    return self.activationFunction(X_with_bias, self.weights)

  def total_loss(self):
    total_loss = np.sum(self.error)
    print(f'Total loss : {total_loss}')
    return total_loss



