import cv2
import numpy as np
import torch
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from PIL import Image
import matplotlib.pyplot as plt

processor = SegformerImageProcessor.from_pretrained("jonathandinu/face-parsing")
model = SegformerForSemanticSegmentation.from_pretrained("jonathandinu/face-parsing")
model.eval() 

def parse_face(cv2_image):
    rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb_image)
    
    inputs = processor(images=img, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    up_logit = torch.nn.functional.interpolate(
        logits, size=img.size[::-1], mode="bilinear", align_corners=False
    )
    labels = up_logit.argmax(dim=1)[0].cpu().numpy()
    return img, labels


def visualize_parsing(img, labels):
    cmap = plt.cm.get_cmap('tab20', np.max(labels)+1)
    plt.figure(figsize=(7,7))
    plt.imshow(img)
    plt.imshow(labels, alpha=0.6, cmap=cmap)
    plt.axis('off')
    plt.show()

def detect_occlusion(labels):
    has_glasses = (labels == 3).any()
    has_nose = (labels == 2).any()
    has_mouth = (labels == 10).any()
    
    occlusions = {
        "sunglasses": has_glasses,
        "nose_visible": has_nose,
        "mouth_visible": has_mouth
    }
    return occlusions
