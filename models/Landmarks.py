import numpy as np
import dlib

def euclidean_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def guess_emotion_from_eyes(eye_landmarks):
    """
    Guess emotion based on eye openness and shape.
    eye_landmarks: List of 12 tuples (6 for each eye)
    """
    left_eye = eye_landmarks[:6]
    right_eye = eye_landmarks[6:]

    def eye_openness(eye):
        return euclidean_distance(eye[1], eye[5]) + euclidean_distance(eye[2], eye[4])

    left_openness = eye_openness(left_eye)
    right_openness = eye_openness(right_eye)

    avg_openness = (left_openness + right_openness) / 2.0

    if avg_openness > 15:
        return "Surprised"
    elif avg_openness < 6:
        return "Sleepy or Angry"
    else:
        return "Neutral or Focused"


def guess_emotion_from_mouth(mouth_landmarks):
    """
    Guess emotion based on mouth shape and openness.
    mouth_landmarks: List of 20 tuples (outer + inner lips)
    """
    outer_lips = mouth_landmarks[:12]
    inner_lips = mouth_landmarks[12:]

    horizontal = euclidean_distance(outer_lips[0], outer_lips[6])  # corner to corner
    vertical = euclidean_distance(outer_lips[3], outer_lips[9])    # top to bottom

    openness_ratio = vertical / horizontal if horizontal > 0 else 0

    if openness_ratio > 0.35:
        return "Surprised or Laughing"
    elif openness_ratio < 0.15:
        return "Neutral or Sad"
    else:
        # Check mouth curvature
        if outer_lips[6][1] < outer_lips[0][1]:
            return "Happy"
        else:
            return "Neutral or Sad"


def obstructed_detection(face_landmarks, obstruction_type):
    """
    face_landmarks: dlib shape object or list of 68 (x, y) tuples
    obstruction_type: 'eyes', 'mouth', or 'none'
    """
    if isinstance(face_landmarks, dlib.full_object_detection):
        coords = [(p.x, p.y) for p in face_landmarks.parts()]
    else:
        coords = face_landmarks

    if obstruction_type == 'eyes':
        mouth = coords[48:68]
        return guess_emotion_from_mouth(mouth)
    elif obstruction_type == 'mouth':
        eyes = coords[36:48]
        return guess_emotion_from_eyes(eyes)
    else:
        return "Use full-face emotion model"
