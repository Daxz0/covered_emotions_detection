import cv2
import dlib
import numpy as np
from matplotlib import pyplot as plt


def detect_face(image_path: str) -> int:
    
    """
    Detects faces in an image and draws rectangles around them.
    
    Returns 0: If there is no face.
    Returns 1: If there is a face.

    :param image_path: Path to the image file
    """
    detector = dlib.get_frontal_face_detector() # type: ignore

    image = cv2.imread(image_path)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    rects = detector(image_rgb, 1)

    if len(rects) < 1:
        return 0

    return 1