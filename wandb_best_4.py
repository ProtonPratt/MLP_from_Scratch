import wandb
import matplotlib.pyplot as plt
import pandas as pd

# Initialize WandB API
api = wandb.Api()

# Define the project
project = 'SMAI_A3-MLP'

# Fetch all runs from the project
runs = api.runs(f"{project}")

# Create a list to store metrics for each run
run_data = []

main_activation = 'relu'
main_batch = 16
main_lr = 0.1
main_hidden_layer = [[16,16]]

activation_functions = ['relu']
batch_sizes = [16]
learning_rates = [0.1]
beta = [0.9]
hidden_layers = [[16,16], [32,32], [64,64], [128,128]]

# hyperparameters
metric = 'hidden_layer_sizes'

# extract the loss graph for each of them and plot them in one graph
losses = {}
for run in runs:
    # Check if specific run name exists (you can remove this condition if you want all runs)
    if 'regression_tuning_1' in run.name:
        if run.config['activation'] in activation_functions and run.config['batch_size'] in \
            batch_sizes and run.config['learning_rate'] in learning_rates and run.config['beta'] in beta and \
            run.config['hidden_layer_sizes'] in hidden_layers:
                
            # Get metrics from the run's summary
            loss = run.history()['val/loss']
            
            print(run.config[metric])
            # print(loss.shape)
            
            # batch_sizes.remove(run.config[metric])
            
            # Append run details to the list
            losses[f'{run.config[metric]}'] = loss
            
            
# plot the graph
# the loss graph is loss vs epoch plot for each of the activation functions
plt.figure(figsize=(10, 6))
for activation, loss in losses.items():
    # print(loss)
    plt.plot(loss, label=activation)

# Customize the plot
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title(f'Loss vs. Epoch for {metric}')
plt.grid(True)
plt.legend()
plt.savefig(f'./assignments/3/images/Loss_vs_epoch_for_{metric}_reg.png')
plt.show()

            