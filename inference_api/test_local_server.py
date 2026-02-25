import requests
import base64
import json

import os
import glob
import time

# Get up to 32 images from the images directory
image_dir = "../images"
# image_dir = "../pipeline-testing/datasets/lfw_flat"

image_paths = glob.glob(os.path.join(image_dir, "*.jpg")) + glob.glob(os.path.join(image_dir, "*.png"))
image_paths = image_paths[:32]

if not image_paths:
    print(f"No images found in {image_dir}")
    exit(1)

b64_strings = []
for ipath in image_paths:
    with open(ipath, "rb") as image_file:
        b64_strings.append(base64.b64encode(image_file.read()).decode("utf-8"))

payload = {
    "inputs": b64_strings,  # Sending an Array of 32 strings
    "parameters": {
        "max_faces": "all"
    }
}

print(f"Sending batch request with {len(b64_strings)} images to local server...")
start_time = time.time()
response = requests.post(
    "http://127.0.0.1:5000/",
    headers={"Content-Type": "application/json"},
    json=payload
)

print(f"Response Code: {response.status_code}")
print(f"Time Taken: {time.time() - start_time:.2f} seconds")

try:
    resp_data = response.json()
    if "batch_faces" in resp_data:
        num_images_processed = len(resp_data["batch_faces"])
        total_faces = sum(len(faces) for faces in resp_data["batch_faces"])
        print(f"Successfully processed {num_images_processed} images.")
        print(f"Found a total of {total_faces} faces across the batch!")
        
        # Print a snippet of the first image's results
        if total_faces > 0:
            print(f"Sample response for Image 1 (truncated):\n{json.dumps(resp_data['batch_faces'][0][:1], indent=2)[:800]} ...")
    else:
        print(f"Response Body:\n{json.dumps(resp_data, indent=2)[:1000]}")
except Exception as e:
    print(response.text)

