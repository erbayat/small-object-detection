import os
from ultralytics import YOLO
import zipfile
import gdown

# Define the directory to save models
MODEL_DIR = "./models/"
os.makedirs(MODEL_DIR, exist_ok=True)

def download_from_google_drive(url, output_path):
    """
    Downloads a file from Google Drive, handling large file confirmation.
    """
    gdown.download(url, output_path, quiet=False, fuzzy=True)

def download_extract_delete_zip_from_gdrive(file_id, output_folder, check_files):
    """
    Downloads, extracts, and deletes a ZIP file if at least one file in the check_files list is missing.
    Args:
        file_id (str): Google Drive file ID.
        output_folder (str): Path to output folder for extraction.
        check_files (list): List of files to check for existence.
    """
    # Check if all specified files exist
    missing_files = [f for f in check_files if not os.path.exists(os.path.join(output_folder, f))]
    
    if not missing_files:
        print(f"All required finetuned models already exist. Skipping download.")
        return

    print(f"The following files are missing and trigger a download: {missing_files}")
    
    # Download the file
    zip_path = "downloaded_models.zip"
    print("Downloading finetuned models...")
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

def install_yolo_model(model_name):
    """
    Downloads and saves a YOLO model.
    Args:
        model_name (str): Name of the YOLO model (e.g., 'yolo11n', 'yolo11s').
    """
    # Change the working directory to MODEL_DIR temporarily
    original_dir = os.getcwd()
    os.chdir(MODEL_DIR)
    try:
        # Check if the file already exists
        file_path = model_name+'.pt'
        if os.path.exists(file_path):
            print(f"The model'{model_name}' already exists. Skipping download.")
            return
        # Load model using YOLO class from ultralytics library
        model = YOLO(model_name)
    finally:
        # Change back to the original working directory
        os.chdir(original_dir)

# URL for the models ZIP file
url = 'https://drive.google.com/file/d/1eUld322Qv5SohOebyEK_Ql864vKPAZwF/view?usp=sharing'

# Install YOLO models
print("Downloading base models...")
install_yolo_model('yolo11n')
install_yolo_model('yolo11s')
install_yolo_model('yolo11m')
install_yolo_model('yolo11l')
install_yolo_model('yolo11x')

# Install fine-tuned versions
finetuned_model_list = ["yolo11n-visdrone.pt","yolo11s-visdrone.pt","yolo11m-visdrone.pt","yolo11l-visdrone.pt","yolo11x-visdrone.pt"]
download_extract_delete_zip_from_gdrive(url, MODEL_DIR, finetuned_model_list)
