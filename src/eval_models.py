import argparse
from ultralytics import YOLO
from pathlib import Path
import time
import csv
import pandas as pd
from sahi import AutoDetectionModel
from sahi.utils.cv import read_image
from sahi.utils.file import download_from_url
from sahi.predict import get_prediction, get_sliced_prediction, predict

MAIN_DIR = "../"
warmup = True

# Function to parse arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description="YOLO model inference on VisDrone dataset.")
    parser.add_argument("--model_name", type=str, default="yolo11n", help="Name of the YOLO model to use (default: yolo11n)")
    parser.add_argument("--dataset_path", type=str, default=r"..\dataset\VisDrone2019-VID-test-dev\sequences\\", help="Path to the dataset folder")
    parser.add_argument("--num_experiments", type=int, default=5, help="Number of experimental runs to perform (default: 5)")
    parser.add_argument("--warmup_runs", type=int, default=2, help="Number of warmup runs (default: 2)")
    parser.add_argument("--is_sahi", type=bool, default=False, help="Whether to use SAHI (default: False)")

    return parser.parse_args()

# Function to set up output directories
def setup_output_directories(base_path, model_name):
    Path(base_path).mkdir(parents=True, exist_ok=True)
    Path(base_path / model_name).mkdir(parents=True, exist_ok=True)

# Function to load class conversions
def get_class_conversions():
    return {
        'v11': {"ignored": 0, 0: 1, "people": 2, 1: 3, 2: 4, "van": 5, 7: 6, "tricycle": 7, "awning-tricycle": 8, 5: 9, 3: 10, "others": 11},
        'visdrone': {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11}
    }


def run_model(model, image_path, is_sahi):
    if not is_sahi:
        return model(image_path, verbose=False)  # Run inference
    else:
        return get_prediction(str(image_path), model)

def add_results_to_list(results, results_list, experiment, category_to_numeric, frame_index ,is_sahi = True):
    if not is_sahi:
        for box in results[0].boxes:
        # Extract bounding box, confidence, and class information
            target_id = "-1"
            bbox = box.xyxy.tolist()
            bbox_left = int(bbox[0][0])
            bbox_top = int(bbox[0][1])
            bbox_width = int(bbox[0][2]) - int(bbox[0][0])
            bbox_height = int(bbox[0][3]) - int(bbox[0][1])
            score = (box.conf).item()
            truncation = "-1"
            occlusion = "-1"
            class_id = category_to_numeric.get(int(box.cls), 11)

            # Save results only once (first run)
            if experiment == 0:
                results_list.append([frame_index, target_id, bbox_left, bbox_top, bbox_width, bbox_height, score, class_id, truncation, occlusion])
    else:
        for detections in results.object_prediction_list:
        # Extract bounding box, confidence, and class information
            target_id = "-1"
            bbox = detections.bbox.to_xywh()
            bbox_left = int(bbox[0])
            bbox_top = int(bbox[1])
            bbox_width = int(bbox[2])
            bbox_height = int(bbox[3])
            score = detections.score.value
            truncation = "-1"
            occlusion = "-1"
            class_id = category_to_numeric.get(int(detections.category.id), 11)

            # Save results only once (first run)
            if experiment == 0:
                results_list.append([frame_index, target_id, bbox_left, bbox_top, bbox_width, bbox_height, score, class_id, truncation, occlusion])

# Function to perform inference on a dataset folder
def process_dataset_folder(model, dataset_folder, category_to_numeric, output_path, num_experiments, warmup_runs, is_sahi):
    image_paths = list(dataset_folder.glob("*.jpg"))
    results_list = []
    latency_data = []

    # Perform warmup runs
    global warmup
    if warmup:
        print(f"Performing {warmup_runs} warmup runs...")
        for _ in range(warmup_runs):
            for image_path in image_paths:
                _ = run_model(model, image_path, is_sahi)  # Run inference but discard the results
        warmup = False

    # Perform experimental runs
    print(f"Starting {num_experiments} experimental runs...")
    for experiment in range(num_experiments):
        experiment_latencies = []
        for counter, image_path in enumerate(image_paths, start=1):
            start_time = time.time()
            results = run_model(model, image_path, is_sahi)  # Run inference
            end_time = time.time()

            latency = end_time - start_time
            experiment_latencies.append(latency)

            frame_index = int(image_path.stem)
            add_results_to_list(results, results_list, experiment, category_to_numeric, frame_index ,is_sahi)

        latency_data.append(experiment_latencies)

    # Save inference results for the first run
    output_file = dataset_folder.name + ".txt"
    with open(output_path / output_file, 'w') as file:
        for result in results_list:
            line = ','.join(map(str, result)) + '\n'
            file.write(line)
    print(f"Results for {dataset_folder.name} saved in {output_file}.")

    # Return latency data for CSV generation
    return latency_data

if __name__ == "__main__":
    args = parse_arguments()

    model_name = args.model_name
    dataset_path = Path(args.dataset_path)
    output_base_path = Path("../results/")
    num_experiments = args.num_experiments
    warmup_runs = args.warmup_runs
    is_sahi = args.is_sahi

    if args.is_sahi:
        model_output_folder = f"{model_name}_sahi"
    else:
        model_output_folder = model_name

    # Load model and class conversions
    if not is_sahi:
        model = YOLO(model_name + ".pt")
    else:
        yolo_model_path = f"../models/{model_name}.pt"
        model = AutoDetectionModel.from_pretrained(
            model_type='yolo11',
            model_path=yolo_model_path,
            confidence_threshold=0.3,
            device="cuda",
        )        
    class_conversions = get_class_conversions()
    setup_output_directories(output_base_path, model_output_folder)

    # Determine class conversion to use
    if model_name.endswith('visdrone'):
        category_to_numeric = class_conversions['visdrone']
    elif model_name.startswith("yolo11"):
        category_to_numeric = class_conversions['v11']

    # Process each dataset folder and collect latency data
    latency_results = []
    for dataset_folder in dataset_path.iterdir():
        if dataset_folder.is_dir():
            latency_data = process_dataset_folder(model, dataset_folder, category_to_numeric, output_base_path / model_output_folder, num_experiments, warmup_runs, is_sahi)
            for frame_idx, latencies in enumerate(zip(*latency_data), start=1):
                avg_latency = sum(latencies) / len(latencies)
                latency_row = {
                    "Dataset Name": 'VisDrone2019-VID-test-dev',
                    "Sequence Name": dataset_folder.name,
                    "Frame Index": frame_idx,
                    "Average Latency": avg_latency,
                    **{f"Ex {i+1}": lat for i, lat in enumerate(latencies)}
                }
                latency_results.append(latency_row)

    # Save latency data to CSV
    csv_file = output_base_path / model_output_folder / "latency_results.csv"
    df = pd.DataFrame(latency_results)
    df.to_csv(csv_file, index=False)
    print(f"Latency results saved to {csv_file}")
