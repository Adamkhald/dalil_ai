import os
import requests
import zipfile
import io

def download_hymenoptera():
    url = "https://download.pytorch.org/tutorial/hymenoptera_data.zip"
    target_dir = os.path.join(os.getcwd(), "data")
    
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        
    print(f"Downloading from {url}...")
    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    print("Extracting to data/...")
    z.extractall(target_dir)
    print("Done!")

if __name__ == "__main__":
    download_hymenoptera()
