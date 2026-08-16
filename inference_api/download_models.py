import os
import urllib.request
import zipfile
import shutil

ZIP_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"

def download_and_extract(url: str, dest_dir: str):
    """Downloads a zip file from a URL and extracts it to the destination directory."""
    if os.path.exists(os.path.join(dest_dir, "det_10g.onnx")) and os.path.exists(os.path.join(dest_dir, "glintr100.onnx")):
        print("✅ Models already exist. Skipping download.")
        return

    os.makedirs(dest_dir, exist_ok=True)
    zip_path = os.path.join(dest_dir, "buffalo_l.zip")
    
    print(f"⬇️ Downloading buffalo_l.zip from {url}...")
    try:
        # Create a custom opener with a standard User-Agent to prevent 403/401 CDN blocking
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response, open(zip_path, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)
            
        print("📦 Extracting required models...")
        REQUIRED_MODELS = ['det_10g.onnx', 'glintr100.onnx']
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for model in REQUIRED_MODELS:
                zip_ref.extract(model, dest_dir)
            
        print("🗑️ Cleaning up zip file...")
        os.remove(zip_path)
        print("🎉 Successfully downloaded and extracted models.")
    except Exception as e:
        print(f"❌ Failed to download models: {e}")
        if os.path.exists(zip_path):
            os.remove(zip_path)

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, "models", "buffalo_l")
    download_and_extract(ZIP_URL, models_dir)
    print("\nAll models are ready!")
