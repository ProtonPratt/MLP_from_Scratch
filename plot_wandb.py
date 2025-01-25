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
        
        # save output.log from the run
        run_file = run.file("output.log")
        run_file.download(replace=True)
        with open("output.log", "r") as f:
            content = f.read()
            
        # spilt
        for i, line in enumerate(content.split("\n")):
            if "Best Activation Function" in line:
                
            
                    

# Convert the list of runs to a pandas DataFrame
df = pd.DataFrame(run_data)

# # Sort the DataFrame by multiple columns for the top 10 runs (based on any metric, e.g., accuracy, precision)
# sorted_df = df.sort_values(by=['precision', 'recall', 'accuracy', 'f1_score'], ascending=[False, False, False, False])

# # Select the top 10 runs
# top_10_runs = sorted_df.head(10)

# # Plot the top 10 runs based on each metric
# metrics = ['precision', 'recall', 'accuracy', 'loss', 'f1_score']
# top_10_runs.set_index('name')[metrics].plot(kind='bar', figsize=(10, 6), colormap='viridis')

# # Set plot labels and title
# plt.title("Top 10 Runs Sorted by Precision, Recall, Accuracy, Loss, and F1 Score")
# plt.xlabel("Run Name")
# plt.ylabel("Metrics")
# plt.xticks(rotation=45, ha="right")

# # Display the plot
# plt.savefig('top_10_runs.png')
# plt.tight_layout()
# plt.show()

# # sort based on precision
# sorted_df = df.sort_values(by=['precision'], ascending=[False])
# top_10_runs = sorted_df.head(10)
# top_10_runs.set_index('name')['precision'].plot(kind='bar', figsize=(10, 6), colormap='viridis')
# plt.title("Top 10 Runs Sorted by Precision")
# plt.xlabel("Run Name")
# plt.ylabel("Precision")
# plt.xticks(rotation=45, ha="right")
# plt.savefig('top_10_runs_precision.png')
# plt.tight_layout()

# # sort based on recall
# sorted_df = df.sort_values(by=['recall'], ascending=[False])
# top_10_runs = sorted_df.head(10)
# top_10_runs.set_index('name')['recall'].plot(kind='bar', figsize=(10, 6), colormap='viridis')
# plt.title("Top 10 Runs Sorted by Recall")
# plt.xlabel("Run Name")
# plt.ylabel("Recall")
# plt.xticks(rotation=45, ha="right")
# plt.savefig('top_10_runs_recall.png')
# plt.tight_layout()

# # sort based on accuracy
# sorted_df = df.sort_values(by=['accuracy'], ascending=[False])
# top_10_runs = sorted_df.head(10)
# top_10_runs.set_index('name')['accuracy'].plot(kind='bar', figsize=(10, 6), colormap='viridis')
# plt.title("Top 10 Runs Sorted by Accuracy")
# plt.xlabel("Run Name")
# plt.ylabel("Accuracy")
# plt.xticks(rotation=45, ha="right")
# plt.savefig('top_10_runs_accuracy.png')
# plt.tight_layout()

# # sort based on loss
# sorted_df = df.sort_values(by=['loss'], ascending=[True])
# top_10_runs = sorted_df.head(10)
# top_10_runs.set_index('name')['loss'].plot(kind='bar', figsize=(10, 6), colormap='viridis')
# plt.title("Top 10 Runs Sorted by Loss")
# plt.xlabel("Run Name")
# plt.ylabel("Loss")
# plt.xticks(rotation=45, ha="right")
# plt.savefig('top_10_runs_loss.png')
# plt.tight_layout()

# # sort based on f1_score
# sorted_df = df.sort_values(by=['f1_score'], ascending=[False])
# top_10_runs = sorted_df.head(10)
# top_10_runs.set_index('name')['f1_score'].plot(kind='bar', figsize=(10, 6), colormap='viridis')
# plt.title("Top 10 Runs Sorted by F1 Score")
# plt.xlabel("Run Name")
# plt.ylabel("F1 Score")
# plt.xticks(rotation=45, ha="right")
# plt.savefig('top_10_runs_f1_score.png')
# plt.tight_layout()
