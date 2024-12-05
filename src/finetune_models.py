import os
from ultralytics import YOLO
import argparse


MAIN_DIR = "./"

# Train the model
def train_yolo_model(model_name, data, epochs, imgsz, suffix, batch):
    """
    Downloads and saves a YOLO model.
    Args:
        model_name (str): Name of the YOLO model (e.g., 'yolov8n', 'yolov8s').
    """
    # Change the working directory to MODEL_DIR temporarily
    original_dir = os.getcwd()
    os.chdir(MAIN_DIR+'models/')
    try:
        # Load model using YOLO class from ultralytics library
        model = YOLO(model_name)
        model.train(data=data, epochs=epochs, imgsz=imgsz, exist_ok=True, name=model_name+suffix, batch=batch)

    finally:
        print("Training completed.")
        # Change back to the original working directory
        os.chdir(original_dir)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLO model with command line arguments")
    parser.add_argument('--model_name', type=str, default='yolo11n.pt', help='Name of the model file')
    parser.add_argument('--data', type=str, default='VisDrone.yaml', help='Path to the data YAML file')
    parser.add_argument('--epochs', type=int, default=400, help='Number of epochs for training')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size for training')
    parser.add_argument('--suffix', type=str, default='_visdrone', help='Suffix for this training run')
    parser.add_argument('--batch', type=int, default=16, help='Batch size for training')

    args = parser.parse_args()
    train_yolo_model(args.model_name, args.data, args.epochs, args.imgsz, args.suffix, args.batch)


