import pandas as pd
import os
from collections import Counter
from PIL import Image
import numpy as np
import pandas as pd


def resolution_adjust(image_path: str, resolution: tuple):
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Skipping {image_path}: {e}")
        return

    resized_image = img.resize(resolution, Image.LANCZOS) # type: ignore
    resized_image.save(image_path)

def image_encoder(image_path: str):
    
    '''
    
    Converts images into numerical format, so the model can read the image.
    
    '''
    
    output = []
    img = Image.open(image_path).convert('RGB')
    arr = np.array(img)
    output.append(arr)
    return np.array(output)


# def prepare_data_set():

    