import os
import pandas as pd
import matplotlib.pyplot as plt

def parse_file(filename):
    """Parses a file and returns a dictionary of metrics."""
    with open(filename, 'r') as f:
        lines = f.readlines()

    metrics = {}
    for line in lines:
        key, value = line.strip().split(': ')
        metrics[key] = float(value)
    return metrics

def plot_metrics(data):
    """Plots the metrics against k and distance metric."""
    for metric in ['Accuracy', 'Macro Precision', 'Micro Precision',
                   'Macro Recall', 'Macro F1 Score', 'Micro F1 Score', 'Specificity']:
        plt.figure()
        for distance_metric, metric_values in data.items():
            plt.plot(metric_values['k'], metric_values[metric], label=distance_metric)
        plt.title(f"{metric} vs. k")
        plt.xlabel("k")
        plt.ylabel(metric)
        plt.legend()
        plt.show()

# Assuming the files are in a directory called "data"
directory = "data"
files = [f for f in os.listdir(directory) if f.startswith('k_')]

data = {}
for file in files:
    metrics = parse_file(os.path.join(directory, file))
    k = int(file.split('_')[1])
    distance_metric = file.split('_')[3]
    if distance_metric not in data:
        data[distance_metric] = {}
    data[distance_metric][k] = metrics

plot_metrics(data)