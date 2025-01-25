import numpy as np
import matplotlib.pyplot as plt
from scipy import odr

# Generate sample data with errors in both x and y
np.random.seed(0)
true_slope = 2
true_intercept = 1
x_true = np.linspace(0, 10, 100)
y_true = true_slope * x_true + true_intercept

# Add noise to both x and y
x = x_true + np.random.normal(0, 1, 100)
y = y_true + np.random.normal(0, 2, 100)

# Define the linear model for ODR
def linear_func(p, x):
    return p[0] * x + p[1]

# Create an ODR model and fit the data
linear_model = odr.Model(linear_func)
data = odr.RealData(x, y)
odr_obj = odr.ODR(data, linear_model, beta0=[1., 1.])
results = odr_obj.run()

# Extract the results
slope_tls, intercept_tls = results.beta
slope_err, intercept_err = results.sd_beta

# Print the results
print(f"TLS Slope: {slope_tls:.4f} ± {slope_err:.4f}")
print(f"TLS Intercept: {intercept_tls:.4f} ± {intercept_err:.4f}")

# Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(x, y, label='Data')
plt.plot(x_true, y_true, 'r-', label='True Line')
plt.plot(x_true, linear_func(results.beta, x_true), 'g--', label='TLS Fit')
plt.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Total Least Squares Regression')
plt.show()