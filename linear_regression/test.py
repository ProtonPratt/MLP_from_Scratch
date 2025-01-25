
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Set up animation
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
import cv2


# class linear_regression:
#     ''' Make sure that atleast X,y have 1 column '''
#     def __init__(self, degree=1, regularisation=None, lamda=0.01, ratio=0.8):
#         self.degree = degree
#         self.weights = None
#         self.bias = None
#         self.type_regularisation = regularisation
#         self.lamda = lamda
#         self.ratio = ratio
        
#     def _prepare(self, X):
#         n_samples, n_features = X.shape
#         X_prepped = np.ones((n_samples, 1))
        
#         for d in range(1, self.degree+1):
#             X_d = X**d
#             X_prepped = np.concatenate((X_prepped, X_d), axis=1)
            
#         return X_prepped
    
#     def regularisation(self):
#         ''' here there are in differentiated terms of regularisation '''
#         if self.type_regularisation is None:
#             return 0
#         if self.type_regularisation == 'l1':
#             return self.lamda * np.sign(self.weights)
#         elif self.type_regularisation == 'l2':
#             return self.lamda * self.weights
#         elif self.type_regularisation == 'elastic':
#             return self.lamda * (self.ratio * self.weights + (1 - self.ratio) * np.sign(self.weights))
#         else:
#             raise ValueError('Unknown regularisation')
        
        
#     def fit(self, X, y, epochs=1000, lr=0.01):
#         ''' X is the feature vector and y is the target vector '''
#         X = np.array(X)
#         y = np.array(y)
        
#         X_prepped = self._prepare(X)
        
#         n_samples, n_features = X_prepped.shape
        
#         self.weights = np.zeros(n_features).reshape(n_features, 1) # Reshape to make it a column vector
#         self.bias = 0
        
#         frames = []
        
#         self._batch_gradient_descent(X_prepped, y, epochs, lr, frames)
            
#         return frames
    
#     def _batch_gradient_descent(self, X, y, epochs=1000, lr=0.01,frames=[]):
#         n_samples, n_features = X.shape
        
#         # Gradient Descent
#         for fr in range(epochs):
#             # compute the h(x)
#             y_pred = np.dot(X, self.weights) + self.bias
            
#             if fr % (epochs//120) == 0:
#                 X_frame = np.linspace(-2, 2, 300).reshape(-1,1)
#                 y_frame = self.predict(X_frame)
#                 metrics = self.evaluate(y, y_pred)
#                 frames.append((X_frame,y_frame,metrics,fr))
            
#             # compute the gradients
#             dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
#             db = (1/n_samples) * np.sum(y_pred - y)
            
#             # regularisation
#             # print(self.regularisation())
#             dw = dw + self.regularisation()
            
#             self.weights -= lr * dw
#             self.bias -= lr * db
        
#     def predict(self, X):
#         X = np.array(X)
#         X_prepped = self._prepare(X)
#         return np.dot(X_prepped, self.weights) + self.bias
    
#     def evaluate(self, y, y_pred):
#         mse = np.mean((y - y_pred) ** 2)
#         variance = np.var(y_pred)
#         std_dev = np.std(y_pred)
        
#         return {
#             'mse': mse,
#             'var': variance,
#             'std': std_dev
#         }
     
    
# def plot_graph_gif(frames, X, y, epochs=1000):
#     fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    
#     # Top-left: Linear Regression Model
#     axs[0, 0].scatter(X, y, color='blue')
#     line, = axs[0, 0].plot([], [], lw=2, color='red')
#     axs[0, 0].set_xlim(-2, 2)
#     axs[0, 0].set_ylim(y.min() - 1, y.max() + 1)
#     axs[0, 0].set_title('Linear Regression')
    
#     # Top-right: MSE
#     mse_line, = axs[0, 1].plot([], [], lw=2, color='orange')
#     axs[0, 1].set_xlim(0, max([f[3] for f in frames]) + 1)
#     axs[0, 1].set_ylim(0, max([f[2]['mse'] for f in frames]) + 1)
#     axs[0, 1].set_title('MSE Over Time')
    
#     # Bottom-left: Variance
#     var_line, = axs[1, 0].plot([], [], lw=2, color='green')
#     # axs[1, 0].set_xlim(0, len(frames))
#     # axs[1, 0].set_ylim(0, max([f[2]['var'] for f in frames]) + 1)
#     axs[1, 0].set_xlim(0, max([f[3] for f in frames]) + 1)
#     axs[1, 0].set_ylim(0, max([f[2]['var'] for f in frames]) + 1)
#     axs[1, 0].set_title('Variance Over Time')
    
#     # Bottom-right: Standard Deviation
#     std_line, = axs[1, 1].plot([], [], lw=2, color='red')
#     axs[1, 1].set_xlim(0, max([f[3] for f in frames]) + 1)
#     axs[1, 1].set_ylim(0, max([f[2]['std'] for f in frames]) + 1)
#     axs[1, 1].set_title('Standard Deviation Over Time')
    
#     def init():
#         line.set_data([], [])
#         mse_line.set_data([], [])
#         var_line.set_data([], [])
#         std_line.set_data([], [])
#         return line, mse_line, var_line, std_line
    
#     def animate(i):
#         # Update Linear Regression Model
#         x = frames[i][0]
#         y_pred = frames[i][1]
#         mse = frames[i][2]['mse']
#         var = frames[i][2]['var']
#         std = frames[i][2]['std']
#         fr = frames[i][3]
        
#         diff = epochs//120
        
#         # print('Epoch:',epoch)
#         print('fr:',fr)
        
#         line.set_data(x, y_pred)
        
#         # Update MSE
#         mse_line.set_data(range(0,fr+1,diff), [f[2]['mse'] for f in frames[:i+1]])
        
#         # Update Variance
#         # var_line.set_data(range(i+1), [f[2]['var'] for f in frames[:i+1]])
#         var_line.set_data(range(0,fr+1,diff), [f[2]['var'] for f in frames[:i+1]])
        
#         # Update Standard Deviation
#         # std_line.set_data(range(i+1), [f[2]['std'] for f in frames[:i+1]])
#         std_line.set_data(range(0,fr+1,diff), [f[2]['std'] for f in frames[:i+1]])
        
#         return line, mse_line, var_line, std_line
    
#     for ax in axs.flat:
#         ax.grid(True)
    
#     anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(frames), interval=100, blit=True)
    
#     anim.save('regression_metrics.gif', writer='imagemagick', fps=20)
    
#     plt.show()

def calculate_mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Function to calculate variance
def calculate_variance(y):
    return np.var(y)

# Function to calculate standard deviation
def calculate_std_dev(y):
    return np.std(y)

from linear_regression import linear_regression, plot_graph_gif

# Load the data
df = pd.read_csv('../../data/external/regularisation.csv')

# shuffle the data
df = df.sample(frac=1).reset_index(drop=True)

X = df['x'].values
y = df['y'].values

X = X.reshape(-1, 1)
y = y.reshape(-1, 1)

# split the data train test val
train_size = int(0.8 * len(X))
validate_size = int(0.1 * len(X))
test_size = len(X) - train_size - validate_size

X_train = X[:train_size]
X_validate = X[train_size:train_size + validate_size]
X_test = X[train_size + validate_size:]

y_train = y[:train_size]
y_validate = y[train_size:train_size + validate_size]
y_test = y[train_size + validate_size:]

# find the best K
mse_k = []

for k in range(1,21):
# Report the MSE, standard deviation and variance metrics on train and test
    print('K:',k)
    model = linear_regression(degree=k, lamda=0.1)
    frames = model.fit(X_train, y_train, epochs=1000, lr=0.02, X_val=X_validate, y_val=y_validate)
    
    # plot gif
    # plot_graph_gif(frames, X, y, epochs=1000, filename='./allgifs/regression_metrics_'+str(k)+'.gif', show=False)

    # print("\nTrain Metrics:")
    y_pred_train = model.predict(X_train)
    # mse_train = calculate_mse(y_train, y_pred_train)
    # variance_train = calculate_variance(y_pred_train)
    # std_dev_train = calculate_std_dev(y_pred_train)
    # print("MSE:", mse_train)
    # print("Variance:", variance_train)
    # print("Standard Deviation:", std_dev_train)

    # print("\nTest Metrics:")
    y_pred_test = model.predict(X_test)
    # mse_test = calculate_mse(y_test, y_pred_test)
    # variance_test = calculate_variance(y_pred_test)
    # std_dev_test = calculate_std_dev(y_pred_test)
    # print("MSE:", mse_test)
    # print("Variance:", variance_test)
    # print("Standard Deviation:", std_dev_test)

    print("\nvalidation Metrics:")
    y_pred_val = model.predict(X_validate)
    mse_val = calculate_mse(y_validate, y_pred_val)
    variance_val = calculate_variance(y_pred_val)
    std_dev_val = calculate_std_dev(y_pred_val)
    print("MSE no:", mse_val)
    print("Variance:", variance_val)
    print("Standard Deviation:", std_dev_val)
    
    X_test_ori = np.linspace(-1.2, 1.2, 100)
    X_test = X_test_ori.reshape(-1,1)
    y_pred = model.predict(X_test)
    # Plot the data and the regression line
    plt.figure(figsize=(15,5))
    
    plt.subplot(1,3,1)
    plt.scatter(X, y, color='blue')
    plt.plot(X_test, y_pred, color='red')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('No Regularisation')
    
    model = linear_regression(degree=k,regularisation='l1' ,lamda=0.8)
    frames = model.fit(X_train, y_train, epochs=1000, lr=0.02, X_val=X_validate, y_val=y_validate)
    y_pred_val = model.predict(X_validate)
    mse_val = calculate_mse(y_validate, y_pred_val)
    variance_val = calculate_variance(y_pred_val)
    std_dev_val = calculate_std_dev(y_pred_val)
    print("MSE l1:", mse_val)
    print("Variance:", variance_val)
    print("Standard Deviation:", std_dev_val)
    
    X_test_ori = np.linspace(-1.2, 1.2, 100)
    X_test = X_test_ori.reshape(-1,1)
    y_pred = model.predict(X_test)
    
    plt.subplot(1,3,2)
    plt.scatter(X, y, color='blue')
    plt.plot(X_test, y_pred, color='red')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('L1 0.8')
    
    model = linear_regression(degree=k,regularisation='l2' ,lamda=0.8)
    frames = model.fit(X_train, y_train, epochs=1000, lr=0.02, X_val=X_validate, y_val=y_validate)
    y_pred_val = model.predict(X_validate)
    mse_val = calculate_mse(y_validate, y_pred_val)
    variance_val = calculate_variance(y_pred_val)
    std_dev_val = calculate_std_dev(y_pred_val)
    print("MSE l2:", mse_val)
    print("Variance:", variance_val)
    print("Standard Deviation:", std_dev_val)
    
    X_test_ori = np.linspace(-1.2, 1.2, 100)
    X_test = X_test_ori.reshape(-1,1)
    y_pred = model.predict(X_test)
    
    plt.subplot(1,3,3)
    plt.scatter(X, y, color='blue')
    plt.plot(X_test, y_pred, color='red')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('L2 0.8')
    
    plt.tight_layout()
    plt.savefig('./k_reg_figs/regression_line_'+str(k)+'.png')
    plt.close()
    
    # plt.show()
    
    mse_k.append((mse_val, k))
    
# Report the best learning rate
best_lr = min(mse_k, key=lambda x: x[0])[1]
print("Best Learning Rate:", best_lr)

# plot the mse vs learning rate
plt.plot([x[1] for x in mse_k], [x[0] for x in mse_k])
plt.xlabel('K')
plt.ylabel('MSE')
plt.savefig('mse_vs_K.png')
plt.show()

# scatter plot of train test and validate
plt.scatter(X_train, y_train, color='blue', label='Train')
plt.scatter(X_test, y_test, color='red', label='Test')
plt.scatter(X_validate, y_validate, color='green', label='Validate')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.savefig('scatter_plot.png')
plt.show()



print('X',X.shape)
print('y',y.shape)

# Do linear regression on the data
model = linear_regression(degree=5, regularisation='l1', lamda=0.1)
frames = model.fit(X_train, y_train, epochs=1000, lr=0.01, X_val=X_validate, y_val=y_validate)

# plot gif
plot_graph_gif(frames, X, y)

X_test_ori = np.linspace(-2, 2, 100)
X_test = X_test_ori.reshape(-1,1)
y_pred = model.predict(X_test)

''' Animation '''
# Set up the figure and axis for animation
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlim(-2, 2)
ax.set_ylim(y.min() - 1, y.max() + 1)
ax.set_xlabel('X')
ax.set_ylabel('y')
ax.set_title('Linear Regression Animation')
scatter, = ax.plot([], [], 'bo', markersize=4)  # Scatter plot of data points
line, = ax.plot([], [], 'r-', lw=2)  # Line plot of predicted values

# Initialization function for animation
def init():
    scatter.set_data([], [])
    line.set_data([], [])
    return scatter, line

# Animation function
def animate(i):
    scatter.set_data(X, y)
    line.set_data(frames[i][0], frames[i][1])
    return scatter, line

# Create the animation
ani = FuncAnimation(fig, animate, frames=len(frames), interval=25, blit=True)

# ani.save('./linear_regression_animation.gif', writer=r'C:\Users\Pratyush Jena\Downloads\ffmpeg-master-latest-win64-gpl\ffmpeg-master-latest-win64-gpl\bin\ffmpeg.exe', fps=20)
ani.save('./linear_regression_animation_new.gif', writer='imagemagick', fps=20)

# Display the animation
plt.show()

# Plot the data and the regression line
plt.scatter(X, y)
plt.plot(X_test, y_pred, color='red')
plt.xlabel('X')
plt.ylabel('y')
plt.show()
