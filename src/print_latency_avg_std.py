import os
import pandas as pd

# Define the results directory
results_dir = "./results/"

# Prepare a list to store the results
results = []

# Define the custom order for sorting
order_priority = {
    "yolo11n": 100, "yolo11s": 200, "yolo11m": 300, "yolo11l": 400, "yolo11x": 500,
}

# Helper function to determine sorting priority
def get_sort_key(subfolder_name):
    sahi_parts = subfolder_name.split("_")
    priority_num = 0
    if len(sahi_parts) > 1:
        priority_num += 10
    visdrone_parts = sahi_parts[0].split("-")
    if len(visdrone_parts) > 1:
        priority_num += 1
    priority_num += order_priority.get(visdrone_parts[0], float('inf'))
    return priority_num

# Iterate through all subfolders in the results directory
for subfolder in os.listdir(results_dir):
    subfolder_path = os.path.join(results_dir, subfolder)
    
    # Check if the subfolder is a directory
    if os.path.isdir(subfolder_path):
        # Construct the file path for latency_results.csv
        file_path = os.path.join(subfolder_path, "latency_results.csv")
        
        # Check if the file exists
        if os.path.exists(file_path):
            # Read the latency results file
            data = pd.read_csv(file_path)
            
            # Calculate the average and standard deviation for Average Latency in milliseconds
            average_latency = data["Average Latency"].mean() * 1_000
            std_latency = data["Average Latency"].std() * 1_000
            
            # Append the results
            results.append({
                "Subfolder": subfolder,
                "Average Latency (ms)": round(average_latency, 1),
                "Standard Deviation (ms)": round(std_latency, 1)
            })

# Convert the results into a DataFrame
results_df = pd.DataFrame(results)

# Sort the DataFrame based on the custom order
results_df["Sort Key"] = results_df["Subfolder"].apply(get_sort_key)
results_df = results_df.sort_values(by="Sort Key").drop(columns=["Sort Key"])

# Display the DataFrame as a markdown table
markdown_table = results_df.to_markdown(index=False)
print(markdown_table)
