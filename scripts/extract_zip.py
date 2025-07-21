import zipfile
from pathlib import Path

def extract_zipped_csv_file(folderPath, fileName):
    zip_path = Path(f"{folderPath}/{fileName}").resolve()
    print("zip_path", zip_path)
    print("folderPath", folderPath)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(f"{folderPath}")