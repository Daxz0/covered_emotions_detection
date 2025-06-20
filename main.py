from models.FaceDetection import detect_face
from models.emotion_detection import detect_emotion
from models.Obstructions import detect_occlusion
from models.Obstructions import visualize_parsing
from models.Obstructions import parse_face
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
import models.Landmarks as landmarks_model

import dlib
from matplotlib import cm

# Load dlib models
frontalface_detector = dlib.get_frontal_face_detector() # type: ignore
landmark_predictor = dlib.shape_predictor('trained_models/landmarks_model.dat') # type: ignore



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

if valid_face == 1:
    img_rgb = np.array(img)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    annotated_image = img_bgr.copy()

    if not occlusions['sunglasses'] and occlusions['nose_visible'] and occlusions['mouth_visible']:
        emotion = detect_emotion(image)
        emotion_text = f"Emotion: {emotion}"
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = frontalface_detector(gray)
        if faces:
            shape = landmark_predictor(gray, faces[0])
            landmarks = [(p.x, p.y) for p in shape.parts()]

            if occlusions['sunglasses'] and occlusions['mouth_visible']:
                # Eyes blocked, mouth available
                mouth_landmarks = landmarks[48:68]
                emotion = landmarks_model.guess_emotion_from_mouth(mouth_landmarks)
                emotion_text = f"Emotion (Mouth): {emotion}"

            elif not occlusions['sunglasses'] and not occlusions['mouth_visible']:
                # Mouth blocked, eyes available
                eye_landmarks = landmarks[36:48]
                emotion = landmarks_model.guess_emotion_from_eyes(eye_landmarks)
                emotion_text = f"Emotion (Eyes): {emotion}"
            else:
                emotion = None
                emotion_text = "Emotion: Not Detected (Too Many Obstructions)"
        else:
            emotion = None
            emotion_text = "Emotion: Not Detected (No Landmarks)"

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


    cv2.imshow("Analysis", blended)
    cv2.waitKey(10000)
    cv2.destroyAllWindows()

else:
    print("There is no valid face in the image.")
