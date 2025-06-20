import os
import numpy as np
import pandas as pd
from PIL import Image

def resolution_adjust(image_path: str, resolution: tuple) -> Image.Image:
    """
    Opens and resizes the image to the given resolution, returns the resized image object.
    """
    try:
        img = Image.open(image_path).convert("L")
        resized_image = img.resize(resolution, Image.LANCZOS) # type: ignore
        return resized_image
    except Exception as e:
        print(f"Skipping {image_path}: {e}")
        return None

def image_encoder(img: Image.Image) -> np.ndarray:
    """
    Converts a PIL image into a flattened numpy array.
    """
    arr = np.array(img)
    return arr.flatten()

def images_to_csv(image_folder: str, output_csv: str, resolution: tuple = (28, 28)):
    """
    Processes all images in a folder: resizes, flattens, and saves them into a CSV.
    """
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    data = []

    for filename in image_files:
        path = os.path.join(image_folder, filename)
        resized_img = resolution_adjust(path, resolution)
        if resized_img is not None:
            flat_array = image_encoder(resized_img)
            data.append(flat_array)

    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"Saved {len(data)} flattened images to {output_csv}")


images_to_csv("data", "output.csv", resolution=(128, 128))