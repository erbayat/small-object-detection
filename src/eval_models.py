import argparse
from ultralytics import YOLO
from pathlib import Path

MAIN_DIR = "../"

# Function to parse arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description="YOLO model inference on VisDrone dataset.")
    parser.add_argument("--model_name", type=str, default="yolo11n", help="Name of the YOLO model to use (default: yolo11n)")
    parser.add_argument("--dataset_path", type=str, default=r".\datasets\VisDrone2019-VID-test-dev\sequences\\", help="Path to the dataset folder")
    return parser.parse_args()

# Function to set up output directories
def setup_output_directories(base_path, model_name):
    Path(base_path).mkdir(parents=True, exist_ok=True)
    Path(base_path + model_name).mkdir(parents=True, exist_ok=True)

# Function to load class conversions
def get_class_conversions():
    return {
        'v11': {"ignored": 0, 0: 1, "people": 2, 1: 3, 2: 4, "van": 5, 7: 6, "tricycle": 7, "awning-tricycle": 8, 5: 9, 3: 10, "others": 11},
        'visdrone': {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11}
    }

# Function to perform inference on a dataset folder
def process_dataset_folder(model, dataset_folder, category_to_numeric, output_path):
    image_paths = list(dataset_folder.glob("*.jpg"))
    results_list = []
    
    for counter, image_path in enumerate(image_paths, start=1):
        results = model(image_path)  # Run inference
        print(f"Processing {image_path.name} in {dataset_folder.name} ({counter}/{len(image_paths)})")

        frame_index = int(image_path.stem)
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

            # Append the results to the list
            results_list.append([frame_index, target_id, bbox_left, bbox_top, bbox_width, bbox_height, score, class_id, truncation, occlusion])
    
    # Save results to a text file
    output_file = dataset_folder.name + ".txt"
    with open(output_path / output_file, 'w') as file:
        for result in results_list:
            line = ','.join(map(str, result)) + '\n'
            file.write(line)

    print(f"Results for {dataset_folder.name} saved in {output_file}.")


if __name__ == "__main__":
    args = parse_arguments()

    model_name = args.model_name
    dataset_path = Path(args.dataset_path)
    output_base_path = Path("results/")
    
    # Load model and class conversions
    model = YOLO(model_name + ".pt")
    class_conversions = get_class_conversions()
    setup_output_directories(output_base_path, model_name)

    # Determine class conversion to use
    if model_name.endswith('visdrone'):
        category_to_numeric = class_conversions['visdrone']
    elif model_name.startswith("yolo11"):
        category_to_numeric = class_conversions['v11']


    # Process each dataset folder
    for dataset_folder in dataset_path.iterdir():
        if dataset_folder.is_dir():
            process_dataset_folder(model, dataset_folder, category_to_numeric, output_base_path / model_name)