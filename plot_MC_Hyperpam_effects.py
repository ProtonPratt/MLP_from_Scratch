''' 1 '''

# import matplotlib.pyplot as plt

# # Placeholder data (replace with your actual loss values)
# activations = ["ReLU", "Sigmoid", "Tanh", "Linear"]
# validation_loss = [0.3, 0.4, 0.35, 0.25]
# test_loss = [0.32, 0.42, 0.37, 0.28]

# # Plotting the graph
# plt.figure(figsize=(10, 6))

# # Validation loss line
# plt.plot(activations, validation_loss, marker='o', label='Validation Loss')

# # Test loss line
# plt.plot(activations, test_loss, marker='s', label='Test Loss')

# # Customize the plot
# plt.xlabel('Activation Function')
# plt.ylabel('Loss')
# plt.title('Validation and Test Loss vs. Activation Function')
# plt.grid(True)
# plt.legend()

# plt.savefig('./assignments/3/images/loss_vs_activation.png')
# plt.show()

'''' 2 '''

# import matplotlib.pyplot as plt

# # Learning rates
# learning_rates = [0.001, 0.01, 0.1, 1]

# # Validation loss
# validation_loss = [0.40869565, 0.40869565, 0.62608695, 0.6173913]

# # Test loss
# test_loss = [0.42982456, 0.42982456, 0.53508772, 0.54385965]

# # Plotting the graph
# plt.figure(figsize=(10, 6))

# # Validation loss line
# plt.plot(learning_rates, validation_loss, marker='o', label='Validation Loss')

# # Test loss line
# plt.plot(learning_rates, test_loss, marker='s', label='Test Loss')

# # Customize the plot
# plt.xlabel('Learning Rate')
# plt.xscale('log')
# plt.ylabel('Loss')
# plt.title('Validation and Test Loss vs. Learning Rate (Sigmoid Activation)')
# plt.grid(True)
# plt.legend()

# plt.savefig('./assignments/3/images/loss_vs_LearningRate.png')
# plt.show()

''' 3 '''

import matplotlib.pyplot as plt

# Batch sizes (including new data)
batch_sizes = [8, 16, 32, 64, 128, 256]

# Validation accuracy (including new data)
validation_accuracy = [0.6347826, 0.6173913, 0.6521739, 0.5913043, 0.4347826, 0.4347826]

# Test accuracy (including new data)
test_accuracy = [0.6140351, 0.5701754, 0.6052632, 0.5526316, 0.4210526, 0.4298246]

# Plotting the graph
plt.figure(figsize=(10, 6))

# Validation accuracy line
plt.plot(batch_sizes, validation_accuracy, marker='o', label='Validation Accuracy')

# Test accuracy line
plt.plot(batch_sizes, test_accuracy, marker='s', label='Test Accuracy')

# Customize the plot
plt.xlabel('Batch Size')
plt.ylabel('Accuracy')
plt.title('Validation and Test Accuracy vs. Batch Size')
plt.grid(True)
plt.legend()

plt.savefig('./assignments/3/images/accuracy_vs_BatchSize.png')
plt.show()