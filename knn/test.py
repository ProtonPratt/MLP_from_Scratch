import os
import re
import matplotlib.pyplot as plt

# Step 1: Reading and Parsing the Files
def parse_metrics_file(filename):
    with open(filename, 'r') as file:
        data = file.read()
    
    # Extract k and distance metric using regex
    k = int(re.search(r'k: (\d+)', data).group(1))
    metric = re.search(r'Distance Metric: (\w+)', data).group(1)
    
    # Extract performance metrics using regex
    metrics = {}
    metrics['accuracy'] = float(re.search(r'Accuracy: ([\d.]+)', data).group(1))
    metrics['macro_precision'] = float(re.search(r'Macro Precision: ([\d.]+)', data).group(1))
    metrics['micro_precision'] = float(re.search(r'Micro Precision: ([\d.]+)', data).group(1))
    metrics['macro_recall'] = float(re.search(r'Macro Recall: ([\d.]+)', data).group(1))
    metrics['macro_f1_score'] = float(re.search(r'Macro F1 Score: ([\d.]+)', data).group(1))
    metrics['micro_f1_score'] = float(re.search(r'Micro F1 Score: ([\d.]+)', data).group(1))
    metrics['specificity'] = float(re.search(r'Specificity: ([\d.]+)', data).group(1))
    metrics['time'] = float(re.search(r'Time: ([\d.]+) seconds', data).group(1))
    
    return k, metric, metrics

def collect_all_metrics(directory):
    all_metrics = {}
    
    for filename in os.listdir(directory):
        if filename.startswith("k_") and filename.endswith(".txt"):  # Adjust according to your file pattern
            filepath = os.path.join(directory, filename)
            k, metric, metrics = parse_metrics_file(filepath)
            
            if metric not in all_metrics:
                all_metrics[metric] = {}
            all_metrics[metric][k] = metrics
    
    return all_metrics

# Step 2: Plotting the Data
def plot_metrics(all_metrics):
    metrics_to_plot = ['accuracy', 'macro_precision', 'micro_precision', 'macro_recall', 'macro_f1_score', 'micro_f1_score', 'specificity', 'time']
    
    for metric_name in metrics_to_plot:
        plt.figure(figsize=(12, 8))
        for distance_metric, data in all_metrics.items():
            k_values = sorted(data.keys())
            y_values = [data[k][metric_name] for k in k_values]
            
            plt.plot(k_values, y_values, marker='o', label=distance_metric)
        
        plt.title(f'{metric_name.replace("_", " ").capitalize()} vs K')
        plt.xlabel('K')
        plt.ylabel(metric_name.replace("_", " ").capitalize())
        plt.legend()
        plt.grid(True)
        plt.show()

# Usage
directory = './knn_results'# Replace with the path to the directory containing your files
all_metrics = collect_all_metrics(directory)
plot_metrics(all_metrics)
