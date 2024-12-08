import argparse
from ultralytics import YOLO
from pathlib import Path
from sahi import AutoDetectionModel
from sahi.utils.cv import read_image
from sahi.utils.file import download_from_url
from sahi.predict import get_prediction, get_sliced_prediction, predict
import matplotlib.pyplot as plt
import cv2

MAIN_DIR = "./"

# Function to parse arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description="YOLO model inference on VisDrone dataset.")
    parser.add_argument("--model_name", type=str, default="yolo11n", help="Name of the base YOLO model to use (default: yolo11n)")
    parser.add_argument("--dataset_path", type=str, default=r"./dataset/VisDrone2019-VID-test-dev/sequences/", help="Path to the dataset folder")
    parser.add_argument("--sequence_path", type=str, default=r"uav0000088_00290_v/", help="Path to the sequence folder")
    parser.add_argument("--frame_index", type=str, default=r"0000005", help="Frame Index")

    return parser.parse_args()





def get_bounding_boxes(model, image_path, is_sahi = True):
    if not is_sahi:
        results = model(image_path, verbose=False)  # Run inference
    else:
        results = get_sliced_prediction(str(image_path), model, slice_height = 512, slice_width = 512, overlap_height_ratio = 0.2, overlap_width_ratio = 0.2, verbose = 0)

    bounding_boxes_with_class = []
    if not is_sahi:
        for box in results[0].boxes:
            bbox = box.xyxy.tolist()
            bounding_boxes_with_class.append([bbox[0][0],bbox[0][1],bbox[0][2],bbox[0][3],(box.conf).item(),model.names[int(box.cls)]])
    else:
        for detections in results.object_prediction_list:
        # Extract bounding box, confidence, and class information
            target_id = "-1"
            bbox = detections.bbox.to_xyxy()
            bounding_boxes_with_class.append([bbox[0],bbox[1],bbox[2],bbox[3],detections.score.value,list(model.category_names)[int(detections.category.id)]])
    return bounding_boxes_with_class


def plot_yolo_results_separately_with_classes(image_path, model_results, labels, colors):

    # Load the image
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.ravel()
    
    for idx, (ax, boxes, label, color) in enumerate(zip(axes, model_results, labels, colors)):
        ax.imshow(image)
        ax.set_title(f"Results from {label}")
        for box in boxes:
            x_min, y_min, x_max, y_max, confidence, class_name = box
            rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, 
                                  linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            ax.text(x_min, y_min - 10, f"{class_name} ({confidence:.2f})", 
                    color=color, fontsize=6, backgroundcolor='black')
        ax.axis('off')
    fig.suptitle(f'Frame Path: {image_path}')
    plt.tight_layout()
    plt.show()

# Function to perform inference on a dataset folder
def get_model_results(base_model,ft_model,base_sahi_model,ft_sahi_model,image_path):
    results = []
    results.append(get_bounding_boxes(base_model,image_path,False))
    results.append(get_bounding_boxes(ft_model,image_path,False))
    results.append(get_bounding_boxes(base_sahi_model,image_path,True))
    results.append(get_bounding_boxes(ft_sahi_model,image_path,True))

    return results

if __name__ == "__main__":

    args = parse_arguments()
    model_name = args.model_name
    dataset_path = Path(args.dataset_path)
    image_path = dataset_path / args.sequence_path / Path(args.frame_index +'.jpg')



    base_model = YOLO(f"./models/{model_name}.pt")
    ft_model = YOLO(f"./models/{model_name}-visdrone.pt")
    base_sahi_model = AutoDetectionModel.from_pretrained(
            model_type='yolo11',
            model_path=f"./models/{model_name}.pt",
            confidence_threshold=0.3,
            device="cuda",
        )   
    ft_sahi_model = AutoDetectionModel.from_pretrained(
            model_type='yolo11',
            model_path=f"./models/{model_name}-visdrone.pt",
            confidence_threshold=0.3,
            device="cuda",
        )   
   


    model_results = get_model_results(base_model,ft_model,base_sahi_model,ft_sahi_model,image_path)

    # Labels for each model
    labels = [f"{model_name}".capitalize()+'-B', f"{model_name}".capitalize()+'-FT', f"{model_name}".capitalize()+'-B++', f"{model_name}".capitalize()+'-FT++']

    # Colors for each model's bounding boxes
    colors = ['red', 'blue', 'green', 'purple']

    # Call the function to plot
    plot_yolo_results_separately_with_classes(image_path, model_results, labels, colors)







