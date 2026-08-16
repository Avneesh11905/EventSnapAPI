import os
import urllib.request

MODELS = {
    "det_10g.onnx": "https://huggingface.co/rdeepc/buffalo_l/resolve/main/det_10g.onnx",
    "glintr100.onnx": "https://huggingface.co/rdeepc/buffalo_l/resolve/main/glintr100.onnx",
}


def download_model(url: str, dest_path: str):
    """Downloads a file from a URL to a local destination if it doesn't already exist."""
    if os.path.exists(dest_path):
        print(f"✅ {os.path.basename(dest_path)} already exists. Skipping.")
        return

    print(f"⬇️ Downloading {os.path.basename(dest_path)} from {url}...")
    try:
        urllib.request.urlretrieve(url, dest_path)
        print(f"🎉 Successfully downloaded {os.path.basename(dest_path)}")
    except Exception as e:
        print(f"❌ Failed to download {os.path.basename(dest_path)}: {e}")
        if os.path.exists(dest_path):
            os.remove(dest_path)


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, "models", "buffalo_l")
    os.makedirs(models_dir, exist_ok=True)

    print("Checking model binaries...")
    for filename, url in MODELS.items():
        destination = os.path.join(models_dir, filename)
        download_model(url, destination)

    print("\nAll models are ready!")
