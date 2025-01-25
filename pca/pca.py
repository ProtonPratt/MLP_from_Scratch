import numpy as np
import pandas as pd

class PCA:
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.components_ = None
        self.mean = None
        self.eigenvalues_ = None

    def fit(self, X):
        # Center the data along the features axis(which will be new coordinate)
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        
        # Calculate the covariance matrix
        # covariance_matrix = np.cov(X_centered, rowvar=False)
        covariance_matrix = np.dot(X_centered.T, X_centered) / X.shape[0]
        
        # Eigen decomposition of the covariance matrix
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        ''' https://numpy.org/doc/stable/reference/generated/numpy.linalg.eig.html '''
        
        
        # Sort eigenvalues and eigenvectors in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        self.eigenvalues_ = eigenvalues[sorted_indices]
        self.components_ = eigenvectors[:, sorted_indices]

        # print("Eigenvalues:", eigenvalues)
        # print( eigenvectors)
        
        # Select a subset of the components if n_components is specified
        if self.n_components is not None:
            self.components_ = self.components_[:, :self.n_components]
            self.eigenvalues_ = self.eigenvalues_[:self.n_components]
            
        # normalize the components if the components are negative make it positive
        # self.components_ = self.components_ / np.linalg.norm(self.components_, axis=0)
        

    def transform(self, X, tol=1e-4):
        # Standardize the data using the mean from the fit
        X_centered = X - self.mean
        
        # Project the data onto the principal components
        X_transformed = np.dot(X_centered, self.components_)
        
        return X_transformed

    def fit_transform(self, X):
        self.fit(X)
        X_transformed = self.transform(X)
        
        # Flip signs of components if necessary (for consistent orientation)
        for i in range(self.components_.shape[1]):
            if np.sum(X_transformed[:, i]) < 0:  # Check if the component is flipped
                X_transformed[:, i] *= -1
        
        return X_transformed
    
    def inverse_transform(self, X_transformed):
        return np.dot(X_transformed, self.components_.T) + self.mean
    
    def CheckPCA(self,X,tol=0.5):
        # Check with the reduced dataset
        X_reduced = self.transform(X)
        X_recovered = self.inverse_transform(X_reduced)
        
        # check dimension
        reduction = X_reduced.shape[1] == self.n_components
        
        # check if the recovered data is close to the original data
        reconstruction_error = np.linalg.norm(X - X_recovered) / np.linalg.norm(X)
        correct_reconstruction = reconstruction_error < tol
        
        print("Reconstruction Error:", reconstruction_error)
        print(reduction, correct_reconstruction)
        return reduction and correct_reconstruction
    
    def explained_variance_ratio(self):
        return self.eigenvalues_ / np.sum(self.eigenvalues_)
    
    
class PCA_sk:
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.components_ = None
        self.mean = None
        self.eigenvalues_ = None

    def fit(self, X):
        # Center the data (subtract the mean)
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        
        # Perform Singular Value Decomposition (SVD)
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        
        # Calculate eigenvalues from singular values
        eigenvalues = S ** 2 / (X.shape[0] - 1)
        
        # If n_components is specified, select only those components
        if self.n_components is not None:
            self.components_ = Vt[:self.n_components]  # Top 'n_components' singular vectors
            self.eigenvalues_ = eigenvalues[:self.n_components]  # Corresponding eigenvalues
        else:
            self.components_ = Vt
            self.eigenvalues_ = eigenvalues

    def transform(self, X):
        # Center the data using the mean from the fit method
        X_centered = X - self.mean
        
        # Project the data onto the principal components
        X_transformed = np.dot(X_centered, self.components_.T)
        
        return X_transformed

    def fit_transform(self, X):
        # Fit the model and transform the data in one step
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X_transformed):
        # Reconstruct the original data from the reduced data
        return np.dot(X_transformed, self.components_) + self.mean
    
    def check_pca(self, X, tol=0.5):
        # Reduce and reconstruct the dataset
        X_reduced = self.transform(X)
        X_recovered = self.inverse_transform(X_reduced)
        
        # Check if the reduction maintains the correct number of dimensions
        reduction = X_reduced.shape[1] == self.n_components
        
        # Calculate reconstruction error (to check how close the reconstructed data is to the original)
        reconstruction_error = np.linalg.norm(X - X_recovered) / np.linalg.norm(X)
        correct_reconstruction = reconstruction_error < tol
        
        print("Reconstruction Error:", reconstruction_error)
        return reduction and correct_reconstruction
    
    def explained_variance_ratio(self):
        # Return the proportion of variance explained by each principal component
        return self.eigenvalues_ / np.sum(self.eigenvalues_)
    
    

# Example usage:
if __name__ == "__main__":
    # Sample data
    np.random.seed(42)
    X_sample = np.random.rand(100, 5)  # 100 samples with 5 features

    # Create PCA instance with 2 components
    pca = PCA(n_components=2)

    # Fit and transform the sample data
    X_pca = pca.fit_transform(X_sample)

    print("Transformed Data Shape:", X_pca.shape)
    print("Principal Components:\n", pca.components_)
    print("Explained Variance:\n", pca.eigenvalues_)