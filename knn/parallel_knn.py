import numpy as np
import multiprocessing as mp
from joblib import Parallel, delayed
from knn import KNNClassifier_R

# Vectorized + Parallelized
class KNNClassifier_P:
    def __init__(self, k, distance_metric='euclidean'):
        self.k = k
        self.distance_metric = distance_metric
        
    def fit(self, X, y):
        self.X = X
        self.y = y
        
    def predict(self, x):
        y_pred = np.zeros(x.shape[0])
        
        # y_pred = Parallel(n_jobs=1)(delayed(self._predict_one)(x[i]) for i in range(x.shape[0]))
        for i in range(x.shape[0]):
            y_pred[i] = self._predict_one(x[i])
        
        return np.array(y_pred)
            
        # return y_pred
    
    def _distance(self, x1, x2):
        if self.distance_metric == 'euclidean':
            return np.sqrt(np.sum((x1[np.newaxis:] - x2) ** 2, axis=1))
        elif self.distance_metric == 'cosine':
            x1 = x1[np.newaxis, :]
            dot_product = np.dot(x1, self.X.T)
            norm_x1 = np.linalg.norm(x1, axis=1)
            norm_x2 = np.linalg.norm(x2, axis=1)
            cosine_sim = dot_product / (norm_x1 * norm_x2.T)
            return 1 - cosine_sim
    
    def _predict_one(self, x):
        distances = self._distance(x, self.X)
        indices = np.argsort(distances)[:self.k]
        labels = self.y[indices]
        # print(labels)
        return np.argmax(np.bincount(labels))
    
    
    
    

    
import time

# test the classifier
np.random.seed(0)
X = np.random.rand(10000, 2)
print(X.shape)
y = (X[:, 0] + X[:, 1] > 1).astype(int)
print(y.shape)
knn = KNNClassifier_R(3)
knn.fit(X, y)
start_time = time.time()
y_pred = knn.predict(X)
end_time = time.time()
print("Accuracy:", np.mean(y == y_pred))

print("Time:", end_time - start_time)

# plot the decision boundary
# h = .01
# x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
# y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
# xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
#                      np.arange(y_min, y_max, h))
# Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
# Z = Z.reshape(xx.shape)
# plt.contourf(xx, yy, Z, alpha=0.8)
# plt.scatter(X[:, 0], X[:, 1], c=y)
# plt.show()
