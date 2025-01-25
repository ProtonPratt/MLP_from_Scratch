import os
import re
import matplotlib.pyplot as plt

# Function to parse the metrics from a file
def parse_metrics(file_path):
    metrics = {}
    with open(file_path, 'r') as file:
        lines = file.readlines()
        
        # Check the total number of lines to ensure "Time" line is within range
        if len(lines) >= 13:
            try:
                print(lines[13])
                # Extract k, distance metric, and all other metrics
                metrics['k'] = int(re.search(r'k:\s*(\d+)', lines[1]).group(1))
                metrics['distance_metric'] = re.search(r'Distance Metric:\s*(\w+)', lines[2]).group(1)
                
                # Extract various performance metrics
                metrics['accuracy'] = float(re.search(r'Accuracy:\s*(\d+\.\d+)', lines[4]).group(1))
                metrics['macro_precision'] = float(re.search(r'Macro Precision:\s*(\d+\.\d+)', lines[5]).group(1))
                metrics['micro_precision'] = float(re.search(r'Micro Precision:\s*(\d+\.\d+)', lines[6]).group(1))
                metrics['macro_recall'] = float(re.search(r'Macro Recall:\s*(\d+\.\d+)', lines[7]).group(1))
                metrics['macro_f1'] = float(re.search(r'Macro F1 Score:\s*(\d+\.\d+)', lines[8]).group(1))
                metrics['micro_f1'] = float(re.search(r'Micro F1 Score:\s*(\d+\.\d+)', lines[9]).group(1))
                metrics['specificity'] = float(re.search(r'Specificity:\s*(\d+\.\d+)', lines[10]).group(1))
                
                # Extract time only if the line exists and matches the pattern
                time_match = re.search(r'Time:\s*(\d+\.\d+)\s*seconds', lines[13])
                if time_match:
                    metrics['time'] = float(time_match.group(1))
                else:
                    metrics['time'] = None  # or you can handle it differently

            except AttributeError as e:
                print(f"Error parsing file {file_path}: {e}")
                metrics = {}  # Reset metrics in case of error
        else:
            print(f"File {file_path} does not contain expected number of lines.")
    
    return metrics

# Function to read all files and parse their metrics
def read_all_metrics(folder_path):
    all_metrics = []
    for file_name in os.listdir(folder_path):
        if file_name.startswith('k_') and file_name.endswith('.txt'):
            file_path = os.path.join(folder_path, file_name)
            metrics = parse_metrics(file_path)
            if metrics:  # Only add if metrics were successfully parsed
                all_metrics.append(metrics)
    return all_metrics

# Function to plot the metrics
def plot_metrics(all_metrics):
    # Separate metrics by distance metric
    metrics_by_distance = {}
    for metric in all_metrics:
        dist_metric = metric['distance_metric']
        if dist_metric not in metrics_by_distance:
            metrics_by_distance[dist_metric] = []
        metrics_by_distance[dist_metric].append(metric)
    
    # Plot each metric for different distance metrics
    for dist_metric, metrics in metrics_by_distance.items():
        metrics.sort(key=lambda x: x['k'])  # Sort by k
        
        ks = [m['k'] for m in metrics]
        accuracy = [m['accuracy'] for m in metrics]
        macro_precision = [m['macro_precision'] for m in metrics]
        micro_precision = [m['micro_precision'] for m in metrics]
        macro_recall = [m['macro_recall'] for m in metrics]
        macro_f1 = [m['macro_f1'] for m in metrics]
        micro_f1 = [m['micro_f1'] for m in metrics]
        specificity = [m['specificity'] for m in metrics]
        times = [m['time'] for m in metrics]
        
        print('time:', times)
        
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(ks, accuracy, label='Accuracy')
        plt.plot(ks, macro_f1, label='Macro F1 Score')
        plt.plot(ks, micro_f1, label='Micro F1 Score')
        plt.xlabel('k')
        plt.ylabel('Score')
        plt.title(f'Accuracy and F1 Scores ({dist_metric})')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 2, 2)
        plt.plot(ks, macro_precision, label='Macro Precision')
        plt.plot(ks, micro_precision, label='Micro Precision')
        plt.xlabel('k')
        plt.ylabel('Precision')
        plt.title(f'Precision ({dist_metric})')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 2, 3)
        plt.plot(ks, macro_recall, label='Macro Recall')
        plt.xlabel('k')
        plt.ylabel('Recall')
        plt.title(f'Recall ({dist_metric})')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 2, 4)
        plt.plot(ks, specificity, label='Specificity')
        plt.plot(ks, times, label='Time (seconds)')
        plt.xlabel('k')
        plt.ylabel('Value')
        plt.title(f'Specificity and Time ({dist_metric})')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'{dist_metric}_metrics.png')
        plt.show()

# Folder where your files are located
folder_path = './knn_results'
all_metrics = read_all_metrics(folder_path)
plot_metrics(all_metrics)
