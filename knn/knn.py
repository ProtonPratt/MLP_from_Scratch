import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from joblib import Parallel, delayed

# Vectorized
class KNNClassifier_V:
    def __init__(self, k, distance_metric='euclidean'):
        self.k = k
        self.distance_metric = distance_metric
        
    def fit(self, X, y):
        self.X = X
        self.y = y
        
    def predict(self, x):
        y_pred = np.zeros(x.shape[0])
        
        for i in range(x.shape[0]):
            y_pred[i] = self._predict_one(x[i])
        
        return np.array(y_pred)
    
    def _distance(self, x1, x2):
        # Distance Metric calculations are vectorized
        if self.distance_metric == 'euclidean':
            return np.sqrt(np.sum((x1[np.newaxis:] - x2) ** 2, axis=1))
        elif self.distance_metric == 'manhattan':
            return np.sum(np.abs(x1[np.newaxis:] - x2), axis=1)
        elif self.distance_metric == 'cosine':
            x1 = x1[np.newaxis, :]
            dot_product = np.dot(x1, x2.T)
            # print(dot_product.shape)
            norm_x1 = np.linalg.norm(x1, axis=1)
            norm_x2 = np.linalg.norm(x2, axis=1)
            cosine_sim = dot_product / (norm_x1 * norm_x2.T)
            return 1 - cosine_sim
        else:
            raise ValueError('Unknown distance metric')
            
    def _predict_one(self, x):
        distances = self._distance(x, self.X)
        distances = distances.reshape(-1)
        indices = np.argsort(distances)[:self.k]
        labels = self.y[indices]
        return np.argmax(np.bincount(labels))
    
    
# Vectorized + Parallelized
class KNNClassifier_P:
    def __init__(self, k, distance_metric='euclidean', n_jobs=8):
        self.k = k
        self.distance_metric = distance_metric
        self.n_jobs = n_jobs
        
    def fit(self, X, y):
        self.X = X
        self.y = y
        
    def predict(self, x):
        y_pred = np.zeros(x.shape[0])
        
        # Parallel Processing
        y_pred = Parallel(n_jobs=self.n_jobs)(delayed(self._predict_one)(x[i]) for i in range(x.shape[0]))
        
        return np.array(y_pred)
    
    def _distance(self, x1, x2):
        # Distance Metric calculations are vectorized
        if self.distance_metric == 'euclidean':
            return np.sqrt(np.sum((x1[np.newaxis:] - x2) ** 2, axis=1))
        elif self.distance_metric == 'manhattan':
            return np.sum(np.abs(x1[np.newaxis:] - x2), axis=1)
        elif self.distance_metric == 'cosine':
            x1 = x1[np.newaxis, :]
            dot_product = np.dot(x1, x2.T)
            # print(dot_product.shape)
            norm_x1 = np.linalg.norm(x1, axis=1)
            norm_x2 = np.linalg.norm(x2, axis=1)
            cosine_sim = dot_product / (norm_x1 * norm_x2.T)
            return 1 - cosine_sim
        else:
            raise ValueError('Unknown distance metric')
            
    
    def _predict_one(self, x):
        distances = self._distance(x, self.X)
        distances = distances.reshape(-1)
        indices = np.argsort(distances)[:self.k]
        labels = self.y[indices]
        return np.argmax(np.bincount(labels))
    
    
# Regular
class KNNClassifier_R:
    def __init__(self, k, distance_metric='euclidean'):
        self.k = k
        self.distance_metric = distance_metric
        
    def fit(self, X, y):
        self.X = X
        self.y = y
        
    def predict(self, x):
        y_pred = np.zeros(x.shape[0])
        
        for i in range(x.shape[0]):
            y_pred[i] = self._predict_one(x[i])
        
        return np.array(y_pred)
    
    def _distance(self, x1, x2):
        # Distance Metric calculations are vectorized
        if self.distance_metric == 'euclidean':
            distance_list = np.zeros(x2.shape[0])
            for i in range(x2.shape[0]):
                diff = x1 - x2[i]
                diff_sq = diff ** 2
                sum = 0
                for j in range(diff_sq.shape[0]):
                    sum += diff_sq[j]
                sqrt = np.sqrt(sum)
                distance_list[i] = sqrt
                
            return distance_list
        
        elif self.distance_metric == 'manhattan':
            # write non vectorized code for manhattan distance
            distance_list = np.zeros(x2.shape[0])
            for i in range(x2.shape[0]):
                diff = x1 - x2[i]
                sum = 0
                for j in range(diff.shape[0]):
                    sum += np.abs(diff[j])
                distance_list[i] = sum
            return distance_list
            # return np.sum(np.abs(x1[np.newaxis:] - x2), axis=1)
        elif self.distance_metric == 'cosine':
            x1 = x1[np.newaxis, :]
            dot_product = np.dot(x1, x2.T)
            # print(dot_product.shape)
            norm_x1 = np.linalg.norm(x1, axis=1)
            norm_x2 = np.linalg.norm(x2, axis=1)
            cosine_sim = dot_product / (norm_x1 * norm_x2.T)
            return 1 - cosine_sim
        else:
            raise ValueError('Unknown distance metric')
            
    def _predict_one(self, x):
        distances = self._distance(x, self.X)
        distances = distances.reshape(-1)
        indices = np.argsort(distances)[:self.k]
        labels = self.y[indices]
        return np.argmax(np.bincount(labels))