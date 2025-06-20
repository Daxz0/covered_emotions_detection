from models.FaceDetection import detect_face
from models.emotion_detection import detect_emotion
from models.Landmarks import detect_occlusion
from models.Landmarks import visualize_parsing
from models.Landmarks import parse_face
import pandas as pd
import os
import numpy as np

import tensorflow as tf
import pandas as pd
import zipfile
import gdown
import cv2
import plotly.express as px
import numpy as np
from pyngrok import ngrok
import seaborn as sns
import matplotlib.pyplot as plt


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    cv2.imshow("Webcam - Press SPACE to capture", frame)
    key = cv2.waitKey(1)

    if key == 27:  # ESC to quit
        cap.release()
        cv2.destroyAllWindows()
        exit()
    elif key == 32:  # SPACE to capture
        image = frame
        break

cap.release()
cv2.destroyAllWindows()


valid_face = detect_face(image)

img, labels = parse_face(image)
occlusions = detect_occlusion(labels)


import cv2
import numpy as np
from matplotlib import cm

# Assuming everything above is already imported and setup...

valid_face = detect_face(image)
img, labels = parse_face(image)
occlusions = detect_occlusion(labels)

if valid_face == 1:
    print("There is a valid face.")
    
    img_rgb = np.array(img)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    annotated_image = img_bgr.copy()

    if not occlusions['sunglasses'] and occlusions['nose_visible'] and occlusions['mouth_visible']:
        emotion = detect_emotion(image)
        emotion_text = f"Emotion: {emotion}"
    else:
        emotion = None
        emotion_text = "Emotion: Not Detected (Occlusion)"

    cmap = cm.get_cmap('tab20', np.max(labels) + 1)
    mask = (cmap(labels)[..., :3] * 255).astype(np.uint8)
    blended = cv2.addWeighted(annotated_image, 0.6, cv2.cvtColor(mask, cv2.COLOR_RGB2BGR), 0.4, 0)

    cv2.putText(blended, emotion_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if emotion else (0, 0, 255), 2)

    y_offset = 60
    for key, value in occlusions.items():
        status = "Yes" if value else "No"
        color = (0, 255, 0) if value else (0, 0, 255)
        cv2.putText(blended, f"{key.replace('_', ' ').title()}: {status}",
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        y_offset += 30
    suggestion = "None"
    if occlusions['sunglasses']:
        suggestion = "→ Future: Mouth-reading Model"
    elif occlusions['nose_visible'] and occlusions['mouth_visible']:
        suggestion = "→ Future: Eye-reading Model"
    else:
        suggestion = "→ Too many obstructions, no emotion can be read"

    cv2.putText(blended, suggestion, (10, y_offset + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("Face Parsing + Emotion Analysis", blended)
    cv2.waitKey(10000)
    cv2.destroyAllWindows()

else:
    print("There is no valid face in the image.")
