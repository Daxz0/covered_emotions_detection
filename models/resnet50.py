import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np

# Emotion map
emotion_map = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral"
}

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet50(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 7)
model.load_state_dict(torch.load("trained_models/resnet50_fer2013.pth", map_location=device))  # Adjust path
model.to(device)
model.eval()

# Transform for input
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Step 1: Open webcam and capture one frame
cap = cv2.VideoCapture(0)
print("[INFO] Press SPACE to capture an image, ESC to exit")

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

# Step 2: Detect face and crop
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

if len(faces) == 0:
    print("[ERROR] No face detected.")
    exit()

for (x, y, w, h) in faces:
    face = image[y:y+h, x:x+w]
    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(face_rgb)

    # Step 3: Predict
    input_tensor = transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        pred_class = torch.argmax(output, dim=1).item()
        emotion = emotion_map[pred_class]
        confidence = F.softmax(output, dim=1)[0][pred_class].item() * 100

    # Step 4: Show result
    print(f"[RESULT] Predicted Emotion: {emotion} ({confidence:.2f}% confidence)")

    # Optional: Display cropped face with prediction
    cv2.putText(face, f"{emotion} ({confidence:.1f}%)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Face Prediction", face)
    cv2.waitKey(3000)
    cv2.destroyAllWindows()
    break
