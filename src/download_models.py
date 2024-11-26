import os
from ultralytics import YOLO
import requests
import zipfile
import gdown

# Define the directory to save models
MODEL_DIR = "../models/"
os.makedirs(MODEL_DIR, exist_ok=True)

def download_from_google_drive(url, output_path):
    """
    Downloads a file from Google Drive, handling large file confirmation.
    """
    gdown.download(url, output_path, quiet=False,fuzzy=True)


def download_extract_delete_zip_from_gdrive(file_id, output_folder):
    # Download the file
    zip_path = "downloaded_models.zip"
    print("Downloading file...")
    download_from_google_drive(file_id, zip_path)

    # Extract the file
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_folder)
        print(f"File extracted to {output_folder}")
    except zipfile.BadZipFile:
        print("The downloaded file is not a valid zip file.")
    finally:
        # Clean up the zip file
        if os.path.exists(zip_path):
            os.remove(zip_path)
            print(f"Deleted zip file: {zip_path}")


def install_yolo_model(model_name):
    """
    Downloads and saves a YOLO model.
    Args:
        model_name (str): Name of the YOLO model (e.g., 'yolo11n', 'yolov11s').
    """
    # Change the working directory to MODEL_DIR temporarily
    original_dir = os.getcwd()
    os.chdir(MODEL_DIR)
    try:
        # Load model using YOLO class from ultralytics library
        model = YOLO(model_name)
        
    finally:
        # Change back to the original working directory
        os.chdir(original_dir)


url = 'https://drive.google.com/file/d/1eUld322Qv5SohOebyEK_Ql864vKPAZwF/view?usp=sharing'

download_extract_delete_zip_from_gdrive(url, MODEL_DIR)

install_yolo_model('yolo11n')
install_yolo_model('yolo11s')
install_yolo_model('yolo11m')
install_yolo_model('yolo11l')
install_yolo_model('yolo11x')


