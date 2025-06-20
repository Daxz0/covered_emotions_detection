from models.FaceDetection import detect_face
import pandas as pd
import os
import numpy as np


def load_data(path: str):
    loaded = np.load(path)
    data = loaded['data']
    labels = loaded['labels']
    return data,labels
    
data,labels = load_data("datasets/emotions_dataset.npz")