import os
import cv2
import dlib
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import matplotlib.pyplot as plt
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

# ====== Label Maps ======
emotion_map = {
    0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy",
    4: "Sad", 5: "Surprise", 6: "Neutral"
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====== Load Emotion Model ======
model = models.resnet50(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 7)
model.load_state_dict(torch.load("trained_models/resnet50_fer2013.pth", map_location=device))
model.to(device)
model.eval()

# ====== Face + Landmark Detection ======
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
dlib_detector = dlib.get_frontal_face_detector()
shape_path = 'shape_predictor_68_face_landmarks.dat'
if not os.path.exists(shape_path):
    import urllib.request
    url = "https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Scholars/Sessions%206%20-%2010%20(Projects)/Project%20-%20Emotion%20Detection/shape_predictor_68_face_landmarks.dat"
    urllib.request.urlretrieve(url, shape_path)
landmark_predictor = dlib.shape_predictor(shape_path)

# ====== Load Face Parser ======
processor = SegformerImageProcessor.from_pretrained("jonathandinu/face-parsing")
parser_model = SegformerForSemanticSegmentation.from_pretrained("jonathandinu/face-parsing")
parser_model.eval()

# ====== Emotion Input Transform ======
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# ====== Helper: Crop to Center Square ======
def crop_to_square_full_face(img, box, padding_ratio=0.3):
    x, y, w, h = box
    cx, cy = x + w // 2, y + h // 2
    face_size = int(max(w, h) * (1 + padding_ratio))  # Add padding
    half_size = face_size // 2

    # Calculate square bounds
    start_x = max(cx - half_size, 0)
    start_y = max(cy - half_size, 0)
    end_x = min(cx + half_size, img.shape[1])
    end_y = min(cy + half_size, img.shape[0])

    # Ensure output is square
    cropped = img[start_y:end_y, start_x:end_x]

    # If not square due to edge cuts, pad it
    h_diff = face_size - (end_y - start_y)
    w_diff = face_size - (end_x - start_x)
    top, bottom = h_diff // 2, h_diff - h_diff // 2
    left, right = w_diff // 2, w_diff - w_diff // 2

    padded = cv2.copyMakeBorder(
        cropped, top, bottom, left, right,
        borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0]
    )
    return padded


# ====== Webcam Capture ======
cap = cv2.VideoCapture(0)
print("[INFO] Press SPACE to capture image or ESC to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        continue
    cv2.imshow("Webcam - Press SPACE", frame)
    key = cv2.waitKey(1)
    if key == 27:
        cap.release()
        cv2.destroyAllWindows()
        exit()
    elif key == 32:
        captured_img = frame.copy()
        break
cap.release()
cv2.destroyAllWindows()

# ====== Face Detection ======
face_img = captured_img.copy()
gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
if len(faces) == 0:
    print("[ERROR] No face detected.")
    exit()

(x, y, w, h) = faces[0]
cv2.rectangle(face_img, (x, y), (x+w, y+h), (0, 255, 0), 2)

# ====== Landmark Detection ======
landmark_img = captured_img.copy()
dlib_faces = dlib_detector(cv2.cvtColor(landmark_img, cv2.COLOR_BGR2RGB), 1)
if len(dlib_faces) > 0:
    landmarks = [(p.x, p.y) for p in landmark_predictor(landmark_img, dlib_faces[0]).parts()]
    for (lx, ly) in landmarks:
        cv2.circle(landmark_img, (lx, ly), 4, (255, 0, 0), -1)

# ====== Emotion Prediction ======
face_crop = crop_to_square_full_face(captured_img, (x, y, w, h))
face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
pil_face = Image.fromarray(face_rgb)

input_tensor = transform(pil_face).unsqueeze(0).to(device)
with torch.no_grad():
    output = model(input_tensor)
    pred_class = torch.argmax(output, dim=1).item()
    emotion = emotion_map[pred_class]
    confidence = F.softmax(output, dim=1)[0][pred_class].item() * 100

# Annotate emotion
prediction_img = face_crop.copy()
cv2.putText(prediction_img, f"{emotion} ({confidence:.1f}%)", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# ====== Face Parsing ======
def parse_face(cv2_image):
    rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb_image)
    inputs = processor(images=img, return_tensors="pt")
    with torch.no_grad():
        outputs = parser_model(**inputs)
    logits = outputs.logits
    up_logit = F.interpolate(logits, size=img.size[::-1], mode="bilinear", align_corners=False)
    labels = up_logit.argmax(dim=1)[0].cpu().numpy()
    return img, labels

def apply_overlay(image_pil, labels_np):
    cmap = plt.cm.get_cmap('tab20', np.max(labels_np) + 1)
    colored_mask = cmap(labels_np)[..., :3]
    overlay = (np.array(image_pil) * 0.4 + colored_mask * 255 * 0.6).astype(np.uint8)
    return overlay

def detect_occlusion(labels):
    has_glasses = (labels == 3).any()
    has_nose = (labels == 2).any()
    has_mouth = (labels == 10).any()
    has_mask = not has_nose and not has_mouth
    emotion_valid = has_nose and has_mouth
    return {
        "sunglasses": has_glasses,
        "mask": has_mask,
        "emotion_valid": emotion_valid
    }

parsed_pil, parsed_labels = parse_face(face_crop)
parsed_img = apply_overlay(parsed_pil, parsed_labels)
occlusion = detect_occlusion(parsed_labels)

# ====== Annotate Parsing Image ======
overlay_label = f"Glasses: {'Yes' if occlusion['sunglasses'] else 'No'} | "
overlay_label += f"Mask: {'Yes' if occlusion['mask'] else 'No'} | "
overlay_label += f"Emotion: {'Valid' if occlusion['emotion_valid'] else 'Unreliable'}"
cv2.putText(parsed_img, overlay_label, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

# ====== Crop All to Square for Display ======
face_disp = crop_to_square_full_face(face_img, (x, y, w, h))
land_disp = crop_to_square_full_face(landmark_img, (x, y, w, h))
pred_disp = prediction_img
pars_disp = parsed_img

# ====== Display All ======
fig, axs = plt.subplots(1, 4, figsize=(22, 6))
fig.suptitle("Face Detection | Landmarks | Emotion | Parsing + Occlusion", fontsize=15)

axs[0].imshow(cv2.cvtColor(face_disp, cv2.COLOR_BGR2RGB))
axs[0].set_title("Face")
axs[0].axis('off')

axs[1].imshow(cv2.cvtColor(land_disp, cv2.COLOR_BGR2RGB))
axs[1].set_title("Landmarks")
axs[1].axis('off')

axs[3].imshow(pars_disp)
axs[3].set_title("Parsing + Occlusion")
axs[3].axis('off')

axs[2].imshow(cv2.cvtColor(pred_disp, cv2.COLOR_BGR2RGB))
axs[2].set_title(f"Emotion: {emotion} ({confidence:.1f}%)")
axs[2].axis('off')

plt.tight_layout()
plt.show()
