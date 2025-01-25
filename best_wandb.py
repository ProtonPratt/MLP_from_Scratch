import wandb

# Initialize WandB API
api = wandb.Api()

# Define the project and (optionally) entity (organization or username)
project = 'SMAI_A3-MLP'
# entity = 'your_entity_name'  # Optional, only needed if part of an organization

# Fetch all runs from the project
runs = api.runs(f"{project}")

# Iterate over runs and print out the name you set for each
for run in runs:
    if 'hyperparam_final-2' in run.name:
        print(run.name)
        print(run.summary)
        

