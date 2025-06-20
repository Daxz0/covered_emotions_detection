import cv2
import dlib
import math
import unittest
import numpy as np
import urllib.request
import os


from scipy.spatial import distance
from matplotlib import pyplot as plt

###Getting the Dlib Shape predictor!

dlibshape_path = 'shape_predictor_68_face_landmarks.dat'

# Download only if not already present
if not os.path.exists(dlibshape_path):
    url = "https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Scholars/Sessions%206%20-%2010%20(Projects)/Project%20-%20Emotion%20Detection/shape_predictor_68_face_landmarks.dat"
    urllib.request.urlretrieve(url, dlibshape_path)
    print("Downloaded shape predictor model.")
else:
    print("Shape predictor model already exists.")
dlibshape_path ='./shape_predictor_68_face_landmarks.dat'


# Load's dlib's pretrained face detector model
frontalface_detector = dlib.get_frontal_face_detector()
#Load the 68 face Landmark file
landmark_predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')

#@title Run this cell to define a helper function for face detection from a url


"""
Returns facial landmarks for the given input image path
"""
def get_landmarks(image_url):
  """
  :type image_url : str
  :rtype image : cv2 object
  :rtype landmarks : list of tuples where each tuple represents
                     the x and y coordinates of facial keypoints
  """

  try:

    #Decodes image address to cv2 object
    url_response = urllib.request.urlopen(image_url)
    img_array = np.array(bytearray(url_response.read()), dtype=np.uint8)
    image = cv2.cvtColor(cv2.imdecode(img_array, -1), cv2.COLOR_BGR2RGB)

  except Exception as e:
    print ("Please check the URL and try again!")
    return None,None

  #Detect the Faces within the image
  faces = frontalface_detector(image, 1)
  if len(faces):
    landmarks = [(p.x, p.y) for p in landmark_predictor(image, faces[0]).parts()]
  else:
    return None,None

  return image,landmarks

#@title Run this cell to define a helper function to visualize landmarks

"""
Display image with its Facial Landmarks
"""
def plot_image_landmarks(image, face_landmarks):
    """
    Display image with its Facial Landmarks.
    :param image: OpenCV image object
    :param face_landmarks: list of (x, y) tuples for facial landmarks
    """
    expected_landmark_count = 68
    image_copy = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)

    # Smaller radius and thickness
    radius = 2  # Small dot
    circle_thickness = -1  # -1 fills the circle

    for (x, y) in face_landmarks:
        cv2.circle(image_copy, (x, y), radius, (255, 0, 0), circle_thickness)

    plt.imshow(image_copy, interpolation='nearest')
    plt.axis('off')
    plt.show()

    # Check for missing landmarks
    if len(face_landmarks) != expected_landmark_count:
        print(f"Warning: Expected 68 landmarks, but found {len(face_landmarks)}.")
    else:
        print("All 68 landmarks successfully detected.")



#Extract the Facial Landmark coordinates
image,landmarks= get_landmarks(input("Enter the URL of the image: ")) #url

#Plot the Facial Landmarks on the face
if landmarks:
  plot_image_landmarks(image,landmarks)
else:
  print ("No Landmarks Detected")
