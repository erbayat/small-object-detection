import os
from ultralytics import YOLO
import requests
import zipfile
import gdown

# Define the directory to save models
MODEL_DIR = "../dataset/"
os.makedirs(MODEL_DIR, exist_ok=True)

def download_from_google_drive(url, output_path):
    """
    Downloads a file from Google Drive, handling large file confirmation.
    """
    gdown.download(url, output_path, quiet=False,fuzzy=True)


def download_extract_delete_zip_from_gdrive(file_id, output_folder, check_folder):

    # Check if the specific folder already exists
    folder_path = os.path.join(output_folder, check_folder)
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        print(f"The folder '{check_folder}' already exists. Dataset is already installed.")
        return
    

    # Download the file
    zip_path = "downloaded_dataset.zip"
    print("Downloading the dataset...")
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




# URL for the dataset
url = 'https://drive.google.com/file/d/1-BEq--FcjshTF1UwUabby_LHhYj41os5/view'

# Check and download the dataset, ensuring 'VisDrone2019-VID-test-dev' exists
download_extract_delete_zip_from_gdrive(url, MODEL_DIR, "VisDrone2019-VID-test-dev")



