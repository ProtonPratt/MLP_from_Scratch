''' Task 2.1 '''

class KernelDensityEstimation:
    def __init__(self, bandwidth=1.0, kernel="gaussian"):
        """
        Initialize KDE with bandwidth and kernel type.
        
        Args:
            bandwidth: Smoothing parameter (default: 1.0)
            kernel: Type of kernel function (default: "gaussian")
        """
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.data = None
        self.n_samples = None
        self.n_features = None
        
    def _gaussian_kernel(self, x):
        """Gaussian kernel function."""
        return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x ** 2)
    
    def _box_kernel(self, x):
        """Box (uniform) kernel function."""
        return np.where(np.abs(x) <= 1, 0.5, 0)
    
    def _triangular_kernel(self, x):
        """Triangular kernel function."""
        return np.where(np.abs(x) <= 1, 1 - np.abs(x), 0)
    
    def _get_kernel(self, x):
        """Select and apply the appropriate kernel function."""
        if self.kernel == "gaussian":
            return self._gaussian_kernel(x)
        elif self.kernel == "box":
            return self._box_kernel(x)
        elif self.kernel == "triangular":
            return self._triangular_kernel(x)
        else:
            raise ValueError("Invalid kernel type")
    
    def fit(self, X):
        """
        Fit the KDE model to the input data.
        
        Args:
            X: Input data of shape (n_samples, n_features)
        """
        self.data = np.array(X)
        self.n_samples, self.n_features = self.data.shape
    
    def _predict_range(self, s_i, n_l, X):
        density = np.zeros(n_l)
        
        for i in range(s_i, s_i + n_l):
            # Calculate distances
            # print(f"i_p: {i}")
            diff = self.data - X[i]
            distances = np.sqrt(np.sum(diff ** 2, axis=1)) / self.bandwidth
            # Apply kernel function and calculate mean
            density[i - s_i] = np.mean(
                self._get_kernel(distances) / (self.bandwidth ** self.n_features)
            )
            
        return density
    
    def predict(self, X):
        if self.data is None:
            raise ValueError("Model must be fitted before making predictions")
        
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
            
        n_points = X.shape[0]
        density = np.zeros(n_points)
        
        for i in range(n_points):
            # Calculate distances
            diff = self.data - X[i]
            distances = np.sqrt(np.sum(diff ** 2, axis=1)) / self.bandwidth
            # Apply kernel function and calculate mean
            density[i] = np.mean(
                self._get_kernel(distances) / (self.bandwidth ** self.n_features)
            )
        
        # # Use parallel processing to compute densities
        # # with Pool(processes=12) as pool:
        # #     # Split the points into chunks and compute densities in parallel
        # #     chunk_size = max(1, n_points // (12 * 4))
        # #     densities = []
        # #     for i in range(0, n_points, chunk_size):
        # #         n_l = min(chunk_size, n_points - i)
        # #         densities.extend(pool.apply_async(self._predict_range, args=(i, n_l, X)).get())
        
        # with Pool(processes=12) as pool:
        #     densities = pool.starmap(self._predict_range, [(i, n_points // 12, X) for i in range(12)])
        
        # # densities = []
        # # for j in range(12):
        # #     num_chunks = 12
        # #     chunk_size = n_points // num_chunks
            
        # #     densities.extend(self._predict_range(j*chunk_size, chunk_size, X))
            
        # densities.extend(self._predict_range(n_points - (n_points % num_chunks), (n_points % num_chunks), X))
            
        # print(f"n_points: {n_points}")
        
        # densities = np.array(densities)
            
        return density
    
    def visualize(self, grid_points=100, data_points=True, s=5, plt_show=True):
        """
        Visualize the density estimation for 2D data.
        
        Args:
            grid_points: Number of points in each dimension for visualization
        """
        if self.data is None:
            raise ValueError("Model must be fitted before visualization")
        if self.n_features != 2:
            raise ValueError("Visualization is only supported for 2D data")
            
        # Create grid of points
        x_min, x_max = self.data[:, 0].min() - 1, self.data[:, 0].max() + 1
        y_min, y_max = self.data[:, 1].min() - 1, self.data[:, 1].max() + 1
        
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, grid_points),
            np.linspace(y_min, y_max, grid_points)
        )
        
        # Evaluate density on grid
        grid_points = np.column_stack((xx.ravel(), yy.ravel()))
        densities = self.predict(grid_points).reshape(xx.shape)
        
        # Create the plot
        plt.figure(figsize=(10, 8))
        
        # Plot the density contours
        plt.contourf(xx, yy, densities, levels=20, cmap='viridis')
        plt.colorbar(label='Density')
        
        # Plot the data points
        if data_points:
            plt.scatter(self.data[:, 0], self.data[:, 1], s=s,
                    color='red', alpha=0.5, label='Data points')
        
        plt.title(f'KDE with {self.kernel} kernel (bandwidth={self.bandwidth})')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        if plt_show:
            plt.show()

# Example usage:
if __name__ == "__main__":
    # Generate sample 2D data
    np.random.seed(42)
    n_samples = 200
    data = np.concatenate([
        np.random.normal(0, 1, (n_samples, 2)),
        np.random.normal(4, 1.5, (n_samples, 2))
    ])
    
    # Create and fit KDE
    kde = KernelDensityEstimation(bandwidth=0.5, kernel="gaussian")
    kde.fit(data)
    
    # Visualize the results
    kde.visualize()