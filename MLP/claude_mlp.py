import numpy as np
from typing import List, Tuple, Callable

class MLPClassifier:
    def __init__(self, hidden_layers: List[int], learning_rate: float = 0.01,
                 activation: str = 'relu', optimizer: str = 'sgd',
                 batch_size: int = 32, epochs: int = 100):
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.activation = activation
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.epochs = epochs
        self.weights = []
        self.biases = []

    def _initialize_parameters(self, input_dim: int, output_dim: int):
        layers = [input_dim] + self.hidden_layers + [output_dim]
        for i in range(1, len(layers)):
            self.weights.append(np.random.randn(layers[i-1], layers[i]) * 0.01)
            self.biases.append(np.zeros((1, layers[i])))

    def _activation_function(self, x: np.ndarray) -> np.ndarray:
        if self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif self.activation == 'tanh':
            return np.tanh(x)
        elif self.activation == 'relu':
            return np.maximum(0, x)
        else:  # linear
            
            return x

    def _activation_derivative(self, x: np.ndarray) -> np.ndarray:
        if self.activation == 'sigmoid':
            s = self._activation_function(x)
            return s * (1 - s)
        elif self.activation == 'tanh':
            return 1 - np.tanh(x)**2
        elif self.activation == 'relu':
            return (x > 0).astype(float)
        else:  # linear
            return np.ones_like(x)

    def forward_propagation(self, X: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        activations = [X]
        zs = []
        for i in range(len(self.weights)):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            zs.append(z)
            a = self._activation_function(z)
            activations.append(a)
        return activations, zs

    def backward_propagation(self, X: np.ndarray, y: np.ndarray, 
                             activations: List[np.ndarray], 
                             zs: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        m = X.shape[0]
        dw = [np.zeros_like(w) for w in self.weights]
        db = [np.zeros_like(b) for b in self.biases]
        
        delta = activations[-1] - y
        dw[-1] = np.dot(activations[-2].T, delta) / m
        db[-1] = np.sum(delta, axis=0, keepdims=True) / m
        
        for l in range(len(self.weights) - 2, -1, -1):
            delta = np.dot(delta, self.weights[l+1].T) * self._activation_derivative(zs[l])
            dw[l] = np.dot(activations[l].T, delta) / m
            db[l] = np.sum(delta, axis=0, keepdims=True) / m
        
        return dw, db

    def _update_parameters(self, dw: List[np.ndarray], db: List[np.ndarray]):
        if self.optimizer == 'sgd':
            for i in range(len(self.weights)):
                self.weights[i] -= self.learning_rate * dw[i]
                self.biases[i] -= self.learning_rate * db[i]
        elif self.optimizer == 'batch':
            # Implementation for Batch Gradient Descent
            pass
        elif self.optimizer == 'mini_batch':
            # Implementation for Mini-Batch Gradient Descent
            pass

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        # One-hot encode the target variable
        y_encoded = np.eye(n_classes)[y.reshape(-1)]
        
        self._initialize_parameters(n_features, n_classes)
        
        for epoch in range(self.epochs):
            for i in range(0, n_samples, self.batch_size):
                X_batch = X[i:i+self.batch_size]
                y_batch = y_encoded[i:i+self.batch_size]
                
                activations, zs = self.forward_propagation(X_batch)
                dw, db = self.backward_propagation(X_batch, y_batch, activations, zs)
                self._update_parameters(dw, db)
            
            # Implement early stopping here

    def predict(self, X: np.ndarray) -> np.ndarray:
        activations, _ = self.forward_propagation(X)
        return np.argmax(activations[-1], axis=1)

    def gradient_check(self, X: np.ndarray, y: np.ndarray, epsilon: float = 1e-7) -> float:
        # Implement gradient checking here
        pass

# Example usage:
# Generate from xor
X_train = np.random.randn(100, 2)
y_train = np.logical_xor(X_train[:, 0] > 0, X_train[:, 1] > 0).astype(int)
X_train, y_train = np.random.randn(100, 10), np.random.randint(0, 5, 100)
mlp = MLPClassifier(hidden_layers=[10, 20, 30, 10], learning_rate=0.01, activation='relu', optimizer='sgd')
mlp.fit(X_train, y_train)
predictions = mlp.predict(X_train)

print("Training Accuracy:", np.mean(predictions == y_train))    
print(y_train)
print(predictions)

# plot the predictions
import matplotlib.pyplot as plt
plt.scatter(range(len(y_train)), y_train, label='True Labels')
plt.scatter(range(len(predictions)), predictions, label='Predictions')
plt.legend()
plt.show()
# Output:
# ![mlp_classifier](mlp_classifier.png)
