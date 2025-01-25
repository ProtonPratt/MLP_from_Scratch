import numpy as np

class PCA:
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.components_ = None
        self.mean = None
        self.eigenvalues_ = None
        self.components_ = None

    def fit(self, X):
        # Center the data along the features axis(which will be new coordinate)
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        
        # Calculate the covariance matrix
        covariance_matrix = np.cov(X_centered, rowvar=False)
        
        # Eigen decomposition of the covariance matrix
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
        ''' https://numpy.org/doc/stable/reference/generated/numpy.linalg.eig.html '''
        
        
        # Sort eigenvalues and eigenvectors in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        self.eigenvalues_ = eigenvalues[sorted_indices]
        self.components_ = eigenvectors[:, sorted_indices]

        print("Eigenvalues:", eigenvalues)
        print( eigenvectors)
        # Select a subset of the components if n_components is specified
        if self.n_components is not None:
            self.components_ = self.components_[:, :self.n_components]
            self.eigenvalues_ = self.eigenvalues_[:self.n_components]

    def transform(self, X):
        # Standardize the data using the mean from the fit
        X_centered = X - self.mean
        
        # Project the data onto the principal components
        X_transformed = np.dot(X_centered, self.components_)
        
        return X_transformed

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

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