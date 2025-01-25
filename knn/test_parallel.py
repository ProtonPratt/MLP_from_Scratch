from joblib import Parallel, delayed
import numpy as np

class KNNClassifier_P:
    def __init__(self, k, distance_metric='euclidean'):
        self.k = k
        self.distance_metric = distance_metric
        
    def fit(self, X, y):
        self.X = X
        self.y = y
        
    def predict(self, x):
        # Parallel Processing
        y_pred = Parallel(n_jobs=10)(delayed(self._predict_one)(x[i]) for i in range(x.shape[0]))
        return np.array(y_pred)
    
    def _distance(self, x1, x2):
        # Distance Metric calculations are vectorized
        if self.distance_metric == 'euclidean':
            return np.sqrt(np.sum((x1[np.newaxis, :] - x2) ** 2, axis=1))
        elif self.distance_metric == 'manhattan':
            return np.sum(np.abs(x1[np.newaxis, :] - x2), axis=1)
        elif self.distance_metric == 'cosine':
            x1 = x1[np.newaxis, :]
            norm_x1 = np.linalg.norm(x1, axis=1, keepdims=True)
            norm_x2 = np.linalg.norm(x2, axis=1)
            dot_product = np.dot(x2, x1.T)
            cosine_sim = dot_product / (norm_x1 * norm_x2)
            cosine_sim = np.clip(cosine_sim, -1.0, 1.0)  # Clip values to handle numerical stability
            return 1 - cosine_sim
        else:
            raise ValueError('Unknown distance metric')
    
    def _predict_one(self, x):
        distances = self._distance(x, self.X)
        indices = np.argsort(distances)[:self.k]
        labels = self.y[indices]
        return np.argmax(np.bincount(labels))

import time

# Test the classifier
np.random.seed(0)
X = np.random.rand(100, 2)
print("X shape:", X.shape)
y = (X[:, 0] + X[:, 1] > 1).astype(int)
print("y shape:", y.shape)

knn = KNNClassifier_P(3, distance_metric='cosine')
knn.fit(X, y)
start_time = time.time()
y_pred = knn.predict(X)
end_time = time.time()

print("Accuracy:", np.mean(y == y_pred))
print("Time:", end_time - start_time)
