import matplotlib.pyplot as plt
import numpy as np

# Data
models = ['Yolo11n', 'Yolo11s', 'Yolo11m', 'Yolo11l', 'Yolo11x']
methods = ['B', 'FT', 'B++', 'FT++']
latency = {
    'B': [28, 28.1, 32.7, 34.6, 44.3],
    'FT': [27.9, 28, 32.5, 34.7, 44.2],
    'B++': [243.2, 246.8, 299.2, 327.9, 466.1],
    'FT++': [258.5, 262.8, 315.2, 344.6, 481.3]
}

std_dev = {
    'B': [10.5, 10.5, 10.7, 10.9, 10.6],
    'FT': [10.4, 10.5, 10.6, 10.7, 10.7],
    'B++': [105.4, 104.3, 127.9, 142.9, 208.5],
    'FT++': [106.8, 106.1, 128.8, 143.1, 208.4]
}

# Plot with unique colors for each method
fig, ax = plt.subplots(figsize=(10, 6))

colors = ['blue', 'green', 'orange', 'red']  # Unique colors for each method

for method, color in zip(methods, colors):
    avg_latency = latency[method]
    std_latency = std_dev[method]
    ax.plot(models, avg_latency, label=f'{method}', marker='o', color=color)
    ax.fill_between(models, 
                    np.array(avg_latency) - np.array(std_latency), 
                    np.array(avg_latency) + np.array(std_latency), 
                    alpha=0.2, color=color)

# Labels and Legend
ax.set_title("Latency Results Across Models", fontsize=14)
ax.set_xlabel("Model Architecture", fontsize=12)
ax.set_ylabel("Average Latency (ms)", fontsize=12)
ax.legend(title="Methods")
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

plt.show()

# Data for accuracy metrics
accuracy_metrics = {
    "AP(.50-.95)-500": [4.27, 16.06, 8.49, 16.94, 6.64, 18.39, 9.64, 19.55, 7.31, 21.12, 10.8, 21.88, 7.96, 20.67, 11.19, 21.6, 7.45, 21.9, 10.78, 21.89],
    "AP(.50)-500": [8.16, 32.48, 17.37, 34.99, 12.24, 37.81, 19.05, 40.79, 13.15, 42.91, 21.28, 45.49, 14.59, 42.33, 21.88, 44.71, 13.47, 44.78, 21.06, 45.71],
    "AP(.75)-500": [3.98, 13.68, 7.38, 14.22, 6.37, 15.24, 8.68, 16.18, 7.17, 17.79, 9.53, 18.2, 7.69, 17.14, 9.98, 17.96, 7.27, 18.3, 9.57, 18.31],
    "AR(.50-.95)-1": [1.64, 7.32, 3.08, 8.22, 2.59, 8.32, 3.73, 8.79, 2.85, 9.16, 4.34, 9.77, 3.21, 9.25, 4.6, 9.83, 3.05, 9.54, 4.33, 9.83],
    "AR(.50-.95)-10": [5.65, 17.33, 9.73, 20.77, 8.11, 20.29, 11.09, 23.35, 8.83, 22.99, 12.25, 25.54, 9.71, 22.86, 12.54, 25.17, 9.24, 24.13, 12.57, 25.83],
    "AR(.50-.95)-100": [6.41, 21.84, 13.38, 25.97, 9.74, 25.43, 14.59, 29.57, 10.43, 28.69, 15.79, 32.14, 11.3, 28.8, 16.15, 31.88, 10.73, 30.2, 16.16, 32.67],
    "AR(.50-.95)-500": [6.41, 21.84, 13.38, 25.97, 9.74, 25.43, 14.59, 29.57, 10.43, 28.69, 15.79, 32.17, 11.3, 28.8, 16.15, 31.91, 10.73, 30.2, 16.16, 32.7],
}

# Extracting data per method
methods_accuracy = ['B', 'FT', 'B++', 'FT++']
data_by_method = {method: [] for method in methods_accuracy}
for i, method in enumerate(methods_accuracy):
    for metric, values in accuracy_metrics.items():
        data_by_method[method].append(values[i::4])

# Plotting accuracy metrics
for metric in accuracy_metrics.keys():
    fig, ax = plt.subplots(figsize=(10, 6))
    metric_values = accuracy_metrics[metric]
    
    for method, color in zip(methods_accuracy, colors):
        values = metric_values[methods_accuracy.index(method)::4]
        ax.plot(models, values, label=f'{method}', marker='o', color=color)

    # Plot labels and title
    ax.set_title(f"{metric} Across Models", fontsize=14)
    ax.set_xlabel("Model Architecture", fontsize=12)
    ax.set_ylabel(metric, fontsize=12)
    ax.legend(title="Methods")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
