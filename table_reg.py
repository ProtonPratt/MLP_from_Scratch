import wandb
import pandas as pd

# Initialize WandB API
api = wandb.Api()

# Define the project
project = 'SMAI_A3-MLP'

# Fetch all runs from the project
runs = api.runs(f"{project}")

# Initialize an empty list to store data for each run
run_data = []

# Iterate over the runs and extract hyperparameters and regression metrics
for run in runs:
    # Extract relevant metrics from each run (MSE, MAE, RMSE, R-squared)
    if 'regression_tuning_2' in run.name:  # Adjust the condition to filter relevant runs
        metrics = run.summary
        val_mse = metrics.get("val/loss", None)
        val_rmse = metrics.get("val/rmse", None)
        val_mae = metrics.get("val/mean_absolute_error", None)
        val_r_squared = metrics.get("val/r_squared", None)
        
        # Extract training metrics for completeness
        train_mse = metrics.get("train/loss", None)
        train_rmse = metrics.get("train/rmse", None)
        train_mae = metrics.get("train/mean_absolute_error", None)
        train_r_squared = metrics.get("train/r_squared", None)

        # Extract hyperparameters for each run
        hyperparams = run.config
        learning_rate = hyperparams.get("learning_rate", None)
        batch_size = hyperparams.get("batch_size", None)
        activation_function = hyperparams.get("activation", None)
        optimizer = hyperparams.get("optimizer", None)
        hidden_layers = hyperparams.get("hidden_layer_sizes", None)

        # Append data for this run to the list
        run_data.append({
            "Run Name": run.name,
            "Learning Rate": learning_rate,
            "Batch Size": batch_size,
            "Activation Function": activation_function,
            "Optimizer": optimizer,
            "Hidden Layers": hidden_layers,
            "Train MSE": train_mse,
            "Train RMSE": train_rmse,
            "Train MAE": train_mae,
            "Train R-Squared": train_r_squared,
            "Val MSE": val_mse,
            "Val RMSE": val_rmse,
            "Val MAE": val_mae,
            "Val R-Squared": val_r_squared  
        })

# Convert the list into a Pandas DataFrame for easy visualization
df = pd.DataFrame(run_data)

# Sort the DataFrame based on Val R-Squared for regression performance
df_sorted = df.sort_values(by="Val R-Squared", ascending=False)

# Display the top 10 runs
print(df_sorted.head(10))

# Optionally, save the DataFrame as a CSV file
df_sorted.to_csv("sorted_regression_metrics_based_on_val_r_squared.csv", index=False)
