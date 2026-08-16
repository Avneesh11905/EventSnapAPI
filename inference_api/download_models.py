import os
import urllib.request
import zipfile
import shutil

MODELS_CONFIG = [
    {
        "url": "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
        "zip_name": "buffalo_l.zip",
        "extract_files": ["det_10g.onnx"],
        "dest_folder": "buffalo_l"
    },
    {
        "url": "https://github.com/deepinsight/insightface/releases/download/v0.7/antelopev2.zip",
        "zip_name": "antelopev2.zip",
        "extract_files": ["glintr100.onnx"],
        "dest_folder": "antelopev2"
    }
]

def download_and_extract():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_base_dir = os.path.join(base_dir, "models")
    
    for config in MODELS_CONFIG:
        dest_dir = os.path.join(models_base_dir, config["dest_folder"])
        
        # Check if all required files already exist
        all_exist = all(os.path.exists(os.path.join(dest_dir, f)) for f in config["extract_files"])
        if all_exist:
            print(f"✅ Models in {config['dest_folder']} already exist. Skipping.")
            continue

        os.makedirs(dest_dir, exist_ok=True)
        zip_path = os.path.join(dest_dir, config["zip_name"])
        
        print(f"⬇️ Downloading {config['zip_name']} from {config['url']}...")
        try:
            req = urllib.request.Request(config["url"], headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response, open(zip_path, 'wb') as out_file:
                shutil.copyfileobj(response, out_file)
                
            print(f"📦 Extracting {config['extract_files']}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                for model in config["extract_files"]:
                    matching_paths = [name for name in zip_ref.namelist() if name.endswith(model)]
                    if not matching_paths:
                        raise ValueError(f"There is no item ending with '{model}' in the archive")
                    
                    # Extract the file (maintains internal zip folder structure)
                    extracted_path = zip_ref.extract(matching_paths[0], dest_dir)
                    
                    # Move it to the root of dest_dir if it was nested
                    final_path = os.path.join(dest_dir, model)
                    if os.path.abspath(extracted_path) != os.path.abspath(final_path):
                        if os.path.exists(final_path):
                            os.remove(final_path)
                        shutil.move(extracted_path, final_path)
                
            print(f"🗑️ Cleaning up {config['zip_name']}...")
            os.remove(zip_path)
            print(f"🎉 Successfully downloaded and extracted {config['dest_folder']} models.")
        except Exception as e:
            print(f"❌ Failed to process {config['zip_name']}: {e}")
            if os.path.exists(zip_path):
                os.remove(zip_path)

if __name__ == "__main__":
    print("Checking model binaries...")
    download_and_extract()
    print("\nAll models are ready!")
