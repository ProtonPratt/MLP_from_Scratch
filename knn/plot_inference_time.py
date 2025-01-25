import matplotlib.pyplot as plt

# Data for execution times
sklearn_times = [0.8067750930786133, 2.693997621536255, 4.950448274612427, 8.575408220291138, 13.647834300994873, 18.721888065338135]
knn_v_times = [1.1080288887023926, 5.457914590835571, 9.485368728637695, 31.583356380462646, 55.19147443771362, 147.86292552947998]
knn_p_times = [2.2716495990753174, 2.450605630874634, 4.334401845932007, 11.643129825592041, 17.668494701385498, 31.79236650466919]

# Dataset sizes
dataset_sizes = [1000, 5000, 10000, 30000, 50000, 114000]

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(dataset_sizes, sklearn_times, label='Scikit-learn KNN')
plt.plot(dataset_sizes, knn_v_times, label='KNNClassifier_V')
plt.plot(dataset_sizes, knn_p_times, label='KNNClassifier_P')

# Add title and labels
plt.title('Execution Time Comparison of KNN Classifiers')
plt.xlabel('Dataset Size')
plt.ylabel('Execution Time (seconds)')
# plt.xscale('log')  # Log scale for better visualization
# plt.yscale('log')  # Log scale for better visualization
plt.legend()

# Show the plot
plt.show()