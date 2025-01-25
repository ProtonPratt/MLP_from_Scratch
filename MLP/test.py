''' MLP class  '''

import numpy as np

class MLP:
    ''' Multi-layer perceptron class '''
    ''' input_size, hidden_layers = [sizes of the hidden layers], output_size, epochs, learning_rate, batch_size, activation_function, loss_function '''
    def __init__(self, input_size, hidden_layers, output_size, epochs = 100, learning_rate = 0.01, early_stopping = False,
                 batch_size = 32, activation_function = 'relu', loss_function = 'mse', optimizer = 'sgd'):
        
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.activation_function = activation_function
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.early_stopping = early_stopping
        
        self.weights = []
        self.biases = []
        self.history = []
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        ''' Initialize weights '''
        layers = [self.input_size] + self.hidden_layers + [self.output_size]
        for i in range(len(layers) - 1):
            self.weights.append(np.random.randn(layers[i], layers[i+1])) # Check the weights initialization
            self.biases.append(np.zeros(layers[i+1]))
    
    # Activation functions       
    def relu(self, x):
        ''' ReLU activation function '''
        return np.maximum(0, x)
    def relu_derivative(self, x):
        ''' ReLU derivative '''
        return np.where(x > 0, 1, 0)
    
    def sigmoid(self, x):
        ''' Sigmoid activation function '''
        return 1 / (1 + np.exp(-x))
    def sigmoid_derivative(self, x):
        ''' Sigmoid derivative '''
        return self.sigmoid(x) * (1 - self.sigmoid(x))
    
    def tanh(self, x):
        ''' Tanh activation function '''
        return np.tanh(x)
    def tanh_derivative(self, x):
        ''' Tanh derivative '''
        return 1 - np.tanh(x)**2
    
    def linear(self, x):
        ''' Linear activation function '''
        return x
    def linear_derivative(self, x):
        ''' Linear derivative '''
        return 1
            
    def _activation(self):
        ''' Activation function '''
        if self.activation_function == 'relu':
            return self.relu, self.relu_derivative
        elif self.activation_function == 'sigmoid':
            return self.sigmoid, self.sigmoid_derivative
        elif self.activation_function == 'tanh':
            return self.tanh, self.tanh_derivative
        elif self.activation_function == 'linear':
            return self.linear, self.linear_derivative
        else:
            raise ValueError('Activation function not supported')
    
    # Optimizers
    def mse(self, y, y_pred):
        ''' Mean squared error '''
        return np.mean((y - y_pred)**2)
    def mse_derivative(self, y, y_pred):
        ''' Mean squared error derivative '''
        return y_pred - y
    
    def cross_entropy(self, y, y_pred):
        ''' Cross entropy loss '''
        return -np.sum(y * np.log(y_pred))
    def cross_entropy_derivative(self, y, y_pred):
        ''' Cross entropy derivative '''
        return (y_pred - y) / (y_pred * (1 - y_pred))
    
    def _loss(self):
        ''' Loss function '''
        if self.loss_function == 'mse':
            return self.mse, self.mse_derivative
        elif self.loss_function == 'cross_entropy':
            return self.cross_entropy, self.cross_entropy_derivative
        else:
            raise ValueError('Loss function not supported')
    
    # Forward pass
    def forward(self, X):
        ''' Forward pass '''
        activations = []
        Z = [X]
        
        activation, _ = self._activation()
        current_activation = X
        
        # Hidden layers
        for i in range(len(self.weights) - 1):
            z = np.dot(current_activation, self.weights[i]) + self.biases[i]
            Z.append(z)
            current_activation = activation(z)
            activations.append(current_activation)
            
        # Output layer
        z = np.dot(current_activation, self.weights[-1]) + self.biases[-1]
        Z.append(z)
        output_activation = activation(z)
        activations.append(output_activation)
        
        return Z, activations
    
    # Backward pass
    def backward(self, X, y, Z, activations):
        ''' Backward pass '''
        grads = {}
        m = y.shape[0]
        # Activation function
        activation_function, activation_derivative = self._activation()
        
        # Loss function
        loss, loss_derivative = self._loss()
        
        # backprop loss
        dA = loss_derivative(y, activations[-1])
        dZ = dA * activation_derivative(Z[-1])
        grads["dW" + str(len(self.weights)-1)] = np.dot(activations[-2].T, dZ) / m
        grads["db" + str(len(self.weights)-1)] = np.sum(dZ, axis=0) / m
        
        # backprop hidden layers
        for i in range(len(self.weights)-2, -1, -1):
            dA = np.dot(dZ, self.weights[i+1].T)
            dZ = dA * activation_derivative(Z[i+1])
            grads["dW" + str(i)] = np.dot(activations[i].T, dZ) / m
            grads["db" + str(i)] = np.sum(dZ, axis=0) / m
            
        return grads
    
    # Update weights
    def update_weights(self, grads):
        ''' Update weights '''
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * grads["dW" + str(i)]
            self.biases[i] -= self.learning_rate * grads["db" + str(i)]
            
    # Train the model
    def fit(self, X, y):
        ''' Train the model '''
        for epoch in range(self.epochs):
            self.optimize(X, y)
            
            if self.early_stopping:
                pass
            
    # Optimize
    def optimize(self, X, y):
        ''' Optimize the model '''
        if self.optimizer == 'sgd':
            self.sgd(X, y)
        elif self.optimizer == 'mini_batch':
            self.mini_batch(X, y)
        elif self.optimizer == 'batch':
            self.batch(X, y)
        else:
            raise ValueError('Optimizer not supported')
    
    def sgd(self, X, y):
        ''' Stochastic gradient descent '''
        for i in range(len(X[0])):
            Z, activations = self.forward(X[i])
            
            # adjust the matrixes for single sample
            for j in range(len(activations)):
                activations[j] = activations[j].reshape(1, -1)
                Z[j] = Z[j].reshape(1, -1)
                
            grads = self.backward(X[i], y[i].reshape(1,-1), Z, activations)
            self.update_weights(grads)
            
    def mini_batch(self, X, y):
        ''' Mini-batch gradient descent '''
        for i in range(0, X.shape[0], self.batch_size):
            Z, activations = self.forward(X[i:i+self.batch_size])
            grads = self.backward(X[i:i+self.batch_size], y[i:i+self.batch_size], Z, activations)
            self.update_weights(grads)
            
    def batch(self, X, y):
        ''' Batch gradient descent '''
        Z, activations = self.forward(X)
        grads = self.backward(X, y, Z, activations)
        self.update_weights(grads)
        
    # Predict
    def predict(self, X):
        ''' Predict '''
        Z, activations = self.forward(X)
        return activations[-1]
        
    
# Test the MLP
mlp = MLP(2, [3, 3], 1, epochs = 100, learning_rate = 0.01, batch_size = 32, activation_function = 'relu', loss_function = 'mse', optimizer = 'sgd')
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# X = np.random.randn(100, 2)
# y = np.random.randint(0, 2, 200).reshape(100, 2)

print(y.shape)

print(X[0])

mlp.fit(X, y)
print(mlp.predict(X))