import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.animation as animation

class linear_regression:
    ''' Make sure that atleast X,y have 1 column '''
    def __init__(self, degree=1, regularisation=None, lamda=0.01, ratio=0.8):
        self.degree = degree
        self.weights = None
        self.bias = None
        self.type_regularisation = regularisation
        self.lamda = lamda
        self.ratio = ratio
        
    def _prepare(self, X):
        n_samples, n_features = X.shape
        X_prepped = np.ones((n_samples, 1))
        
        for d in range(1, self.degree+1):
            X_d = X**d
            X_prepped = np.concatenate((X_prepped, X_d), axis=1)
            
        return X_prepped
    
    def regularisation(self):
        ''' here there are in differentiated terms of regularisation '''
        if self.type_regularisation is None:
            return 0
        if self.type_regularisation == 'l1':
            return self.lamda * np.sign(self.weights)
        elif self.type_regularisation == 'l2':
            return self.lamda * self.weights
        elif self.type_regularisation == 'elastic':
            return self.lamda * (self.ratio * self.weights + (1 - self.ratio) * np.sign(self.weights))
        else:
            raise ValueError('Unknown regularisation')
        
        
    def fit(self, X, y, epochs=1000, lr=0.01, X_val=None, y_val=None):
        ''' X is the feature vector and y is the target vector '''
        self.X_val = X_val
        self.y_val = y_val
        
        X = np.array(X)
        y = np.array(y)
        
        X_prepped = self._prepare(X)
        
        n_samples, n_features = X_prepped.shape
        
        self.weights = np.zeros(n_features).reshape(n_features, 1) # Reshape to make it a column vector
        self.bias = 0
        
        frames = []
        
        self._batch_gradient_descent(X_prepped, y, epochs, lr, frames)
            
        return frames
    
    def _batch_gradient_descent(self, X, y, epochs=1000, lr=0.01,frames=[]):
        n_samples, n_features = X.shape
        
        # Gradient Descent
        for fr in range(epochs):
            # compute the h(x)
            y_pred = np.dot(X, self.weights) + self.bias
            
            if fr % (epochs//120) == 0:
                # sort
                # self.X_val = np.sort(self.X_val)
                X_frame = self.X_val.reshape(-1,1)
                y_frame = self.predict(self.X_val)
                metrics = self.evaluate(self.y_val, y_frame)
                X_frame = np.linspace(-2, 2, 300).reshape(-1,1)
                y_frame = self.predict(X_frame)
                frames.append((X_frame,y_frame,metrics,fr))
            
            # if fr % (epochs//120) == 0:
            #     X_frame = np.linspace(-2, 2, 300).reshape(-1,1)
            #     y_frame = self.predict(X_frame)
            #     metrics = self.evaluate(y, y_pred)
            #     frames.append((X_frame,y_frame,metrics,fr))
            
            # compute the gradients
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)
            
            # regularisation
            # print(self.regularisation())
            dw = dw + self.regularisation()
            
            self.weights -= lr * dw
            self.bias -= lr * db
        
    def predict(self, X):
        X = np.array(X)
        X_prepped = self._prepare(X)
        return np.dot(X_prepped, self.weights) + self.bias
    
    def evaluate(self, y, y_pred):
        mse = np.mean((y - y_pred) ** 2)
        variance = np.var(y - y_pred)
        std_dev = np.std(y - y_pred)
        
        return {
            'mse': mse,
            'var': variance,
            'std': std_dev
        }
    
    def save_weights(self):
        # write both weight and bias to a file
        open('weights.txt', 'w').write(f'{self.weights}\n{self.bias}')
        
        return self.weights, self.bias
    
def plot_graph_gif(frames, X, y, epochs=1000, show=True, filename='regression_metrics.gif'):
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    
    # Top-left: Linear Regression Model
    axs[0, 0].scatter(X, y, color='blue')
    line, = axs[0, 0].plot([], [], lw=2, color='red')
    axs[0, 0].set_xlim(-2, 2)
    axs[0, 0].set_ylim(y.min() - 1, y.max() + 1)
    axs[0, 0].set_title('Linear Regression')
    
    # Top-right: MSE
    mse_line, = axs[0, 1].plot([], [], lw=2, color='orange')
    axs[0, 1].set_xlim(0, max([f[3] for f in frames]) + 1)
    axs[0, 1].set_ylim(0, max([f[2]['mse'] for f in frames]) + 1)
    axs[0, 1].set_title('MSE Over Time')
    
    # Bottom-left: Variance
    var_line, = axs[1, 0].plot([], [], lw=2, color='green')
    # axs[1, 0].set_xlim(0, len(frames))
    # axs[1, 0].set_ylim(0, max([f[2]['var'] for f in frames]) + 1)
    axs[1, 0].set_xlim(0, max([f[3] for f in frames]) + 1)
    axs[1, 0].set_ylim(0, max([f[2]['var'] for f in frames]))
    axs[1, 0].set_title('Variance Over Time')
    
    # Bottom-right: Standard Deviation
    std_line, = axs[1, 1].plot([], [], lw=2, color='red')
    axs[1, 1].set_xlim(0, max([f[3] for f in frames]) + 1)
    axs[1, 1].set_ylim(0, max([f[2]['std'] for f in frames]))
    axs[1, 1].set_title('Standard Deviation Over Time')
    
    def init():
        line.set_data([], [])
        mse_line.set_data([], [])
        var_line.set_data([], [])
        std_line.set_data([], [])
        return line, mse_line, var_line, std_line
    
    
    def animate(i):
        # Update Linear Regression Model
        x = frames[i][0]
        y_pred = frames[i][1]
        mse = frames[i][2]['mse']
        var = frames[i][2]['var']
        std = frames[i][2]['std']
        fr = frames[i][3]
        
        diff = epochs//120
        
        # print('Epoch:',epoch)
        print('fr:',fr)
        
        line.set_data(x, y_pred)
        
        # Update MSE
        mse_line.set_data(range(0,fr+1,diff), [f[2]['mse'] for f in frames[:i+1]])
        
        # Update Variance
        # var_line.set_data(range(i+1), [f[2]['var'] for f in frames[:i+1]])
        var_line.set_data(range(0,fr+1,diff), [f[2]['var'] for f in frames[:i+1]])
        
        # Update Standard Deviation
        # std_line.set_data(range(i+1), [f[2]['std'] for f in frames[:i+1]])
        std_line.set_data(range(0,fr+1,diff), [f[2]['std'] for f in frames[:i+1]])
        
        return line, mse_line, var_line, std_line
    
    for ax in axs.flat:
        ax.grid(True)
    
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(frames), interval=100, blit=True)
    
    
    anim.save(filename, writer='imagemagick', fps=20)
    
    if show:
        plt.show()
    

