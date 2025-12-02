import os
import requests
import zipfile
import shutil
from colorama import Fore, Style, init
import gdown

# Initialize colorama
init(autoreset=True)

# Backup/Mirror links if official fails (Trying a known mirror or using gdown if a Drive link is found)
# Since we couldn't find a direct stable Drive link, we will try the official one again with a robust method, 
# and if it fails, we'll simulate the 'Panic' class using a subset of 'Violence' or a placeholder 
# so the user can proceed with the pipeline while manually fixing the data later.

AVENUE_DATASET_URL = "http://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/Avenue_Dataset.zip"
DATA_DIR = os.path.join(os.getcwd(), "Avenue Dataset")
ZIP_FILE = os.path.join(os.getcwd(), "Avenue_Dataset.zip")

def download_file(url, dest_path):
    """Downloads a file from a URL to a destination path."""
    if os.path.exists(dest_path):
        print(f"{Fore.YELLOW}File already exists: {dest_path}")
        return

    print(f"{Fore.CYAN}Downloading {url}...")
    try:
        # User-Agent to avoid some server blocks
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, stream=True, timeout=60)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192
        
        with open(dest_path, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=block_size):
                f.write(chunk)
                downloaded += len(chunk)
                # Simple progress
                if total_size > 0:
                    percent = int(50 * downloaded / total_size)
                    print(f"\r[{'=' * percent}{' ' * (50 - percent)}] {downloaded}/{total_size} bytes", end='')
        print(f"\n{Fore.GREEN}Download complete: {dest_path}")
    except Exception as e:
        print(f"\n{Fore.RED}Failed to download {url}: {e}")

def extract_zip(zip_path, extract_to):
    """Extracts a zip file to a destination directory."""
    print(f"{Fore.CYAN}Extracting {zip_path}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"{Fore.GREEN}Extraction complete.")
    except Exception as e:
        print(f"{Fore.RED}Failed to extract {zip_path}: {e}")

def prepare_avenue_dataset():
    """Downloads and prepares the Avenue dataset for the 'Panic' class."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)

    # Attempt download
    if not os.path.exists(ZIP_FILE):
        download_file(AVENUE_DATASET_URL, ZIP_FILE)

    if os.path.exists(ZIP_FILE):
        # Check if valid zip
        try:
            extract_zip(ZIP_FILE, os.getcwd())
        except zipfile.BadZipFile:
            print(f"{Fore.RED}Downloaded file is not a valid zip. Deleting it.")
            os.remove(ZIP_FILE)
    else:
        print(f"{Fore.RED}Zip file not found. Download might have failed.")

    # Fallback/Check
    if os.path.exists(os.path.join(os.getcwd(), "Avenue Dataset", "training_videos")):
         print(f"{Fore.GREEN}Avenue dataset structure seems correct.")
    else:
         print(f"{Fore.YELLOW}Warning: Avenue dataset download failed or structure is unexpected.")
         print(f"{Fore.YELLOW}Manual Step: Download 'Avenue_Dataset.zip' from {AVENUE_DATASET_URL} and place it in project root.")

if __name__ == "__main__":
    prepare_avenue_dataset()
