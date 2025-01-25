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

# Iterate over the runs and extract hyperparameters and metrics
for run in runs:
    # Extract relevant metrics from each run (Precision, Recall, Accuracy, Loss, F1 score)
    if 'hyperparam_MC_3_test' in run.name:
        metrics = run.summary
        val_accuracy = metrics.get("val/accuracy", None)
        val_loss = metrics.get("val/loss", None)
        val_precision = metrics.get("val/precision", None)
        val_recall = metrics.get("val/recall", None)
        val_f1_score = metrics.get("val/f1_score", None)

        # Extract training metrics for completeness
        train_accuracy = metrics.get("train/accuracy", None)
        train_loss = metrics.get("train/loss", None)
        train_precision = metrics.get("train/precision", None)
        train_recall = metrics.get("train/recall", None)
        train_f1_score = metrics.get("train/f1_score", None)

        # Extract hyperparameters for each run
        hyperparams = run.config
        learning_rate = hyperparams.get("learning_rate", None)
        batch_size = hyperparams.get("batch_size", None)
        activation_function = hyperparams.get("activation_function", None)
        optimizer = hyperparams.get("optimizer", None)
        hidden_layers = hyperparams.get("hidden_layers", None)

        # Append data for this run to the list
        run_data.append({
            "Run Name": run.name,
            "Learning Rate": learning_rate,
            "Batch Size": batch_size,
            "Activation Function": activation_function,
            "Optimizer": optimizer,
            "Hidden Layers": hidden_layers,
            "Train Accuracy": train_accuracy,
            "Train Loss": train_loss,
            "Train Precision": train_precision,
            "Train Recall": train_recall,
            "Train F1 Score": train_f1_score,
            "Val Accuracy": val_accuracy,
            "Val Loss": val_loss,
            "Val Precision": val_precision,
            "Val Recall": val_recall,
            "Val F1 Score": val_f1_score
        })

# Convert the list into a Pandas DataFrame for easy visualization
df = pd.DataFrame(run_data)

# Sort the DataFrame based on val/accuracy
df_sorted = df.sort_values(by="Val Accuracy", ascending=False)

# Display the top 10 runs
print(df_sorted.head(10))

# Optionally, save the DataFrame as a CSV file
df_sorted.to_csv("sorted_hyperparameter_metrics_based_on_val_accuracy.csv", index=False)