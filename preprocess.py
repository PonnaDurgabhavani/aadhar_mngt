import os
import cv2
import pandas as pd
from tqdm import tqdm
from PIL import Image
import numpy as np

# --- PATH CONFIG ---
DATASET_PATH = "aadhar_mngt"     # change if your folder name is different
OUTPUT_PATH = "processed_dataset"
os.makedirs(OUTPUT_PATH, exist_ok=True)

# --- LIST TO STORE METADATA ---
metadata = []

# --- PREPROCESSING FUNCTION ---
def preprocess_image(image_path, save_path):
    img = cv2.imread(image_path)

    # Step 1: Resize all images to 640x640
    img_resized = cv2.resize(img, (640, 640))

    # Step 2: Convert to grayscale
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

    # Step 3: Apply Gaussian Blur to reduce noise
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    # Step 4: Normalize pixel values (0-1 range)
    normalized = blur / 255.0

    # Step 5: Save processed image
    cv2.imwrite(save_path, (normalized * 255).astype(np.uint8))

# --- LOOP THROUGH DATASET ---
for folder in os.listdir(DATASET_PATH):
    folder_path = os.path.join(DATASET_PATH, folder)

    # ✅ Skip non-folder items (like LICENSE, .git, etc.)
    if not os.path.isdir(folder_path):
        continue

    save_folder = os.path.join(OUTPUT_PATH, folder)
    os.makedirs(save_folder, exist_ok=True)


    for img_file in tqdm(os.listdir(folder_path), desc=f"Processing {folder}"):
        if img_file.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(folder_path, img_file)
            save_path = os.path.join(save_folder, img_file)

            # Preprocess image
            preprocess_image(img_path, save_path)

            # Extract metadata
            img = Image.open(img_path)
            metadata.append({
                "filename": img_file,
                "document_type": folder,
                "format": img.format,
                "mode": img.mode,
                "width": img.width,
                "height": img.height,
                "size_kb": round(os.path.getsize(img_path) / 1024, 2)
            })

# --- SAVE METADATA TO CSV ---
metadata_df = pd.DataFrame(metadata)
metadata_df.to_csv("metadata.csv", index=False)

print("\n✅ Preprocessing complete!")
print(f"Processed images saved to: {OUTPUT_PATH}")
print("Metadata saved to: metadata.csv")
