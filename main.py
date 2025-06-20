from models.FaceDetection import detect_face
import pandas as pd
import os
import numpy as np

from models.FaceDetection import detect_face

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

def launch_website():
  try:
    if ngrok.get_tunnels():
      ngrok.kill()
    tunnel = ngrok.connect()

    print("Click this link to try your web app:")
    print(tunnel.public_url)

    !streamlit run --server.port 80 app.py >/dev/null # Connect to the URL through Port 80 (>/dev/null hides outputs)

  except KeyboardInterrupt:
    ngrok.kill()


input_path = "image.jpg"


# if detect_face(input_path):
    

# else:
#     print("No face was found in the image.")

 