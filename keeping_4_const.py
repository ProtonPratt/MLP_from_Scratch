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

# Iterate over runs and extract relevant metrics
for run in runs:
    # Check if specific run name exists (you can remove this condition if you want all runs)
    if 'hyperparam_final-2' in run.name:
        # Get metrics from the run's summary
        precision = run.summary.get('precision', 0)
        recall = run.summary.get('recall', 0)
        accuracy = run.summary.get('accuracy', 0)
        loss = run.summary.get('loss', float('inf'))  # In case the loss is not available, set it to infinity
        f1_score = run.summary.get('f1_score', 0)
        
        # Append run details to the list
        run_data.append({
            'name': run.name,
            'precision': precision,
            'recall': recall,
            'accuracy': accuracy,
            'loss': loss,
            'f1_score': f1_score
        })

# Convert the list of runs to a pandas DataFrame
df = pd.DataFrame(run_data)

# select 4 activation function and plot the 