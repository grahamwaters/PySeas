import requests
import cv2
import numpy as np
import os
from tqdm import tqdm
from PIL import Image
from io import BytesIO
from datetime import datetime
from ratelimit import limits, sleep_and_retry
import matplotlib.pyplot as plt
import pandas as pd
import time
import random
import glob
import shutil
import imutils

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image

# Import the functions from phase_one module
from phase_one import (download_image, resize_image_to_standard_height, split_image_into_panels, detect_horizon_line, align_horizon_line, mse, check_unusual_panels, stitch_aligned_images, load_skipped_buoys)

MODEL_PATH = 'models/gen3_keras/keras_model.h5'
IMAGE_SIZE = (224, 224)
CLASS_NAMES = ['Sunset', 'Storms', 'Normal', 'Object In View']
output_dir = "scripts/output"
collecting_all = True
buoy_list = pd.read_csv("scripts/working_buoys.csv")

# Classification Functions
def classify_image(image, model, image_size=IMAGE_SIZE, class_names=CLASS_NAMES):
    if not isinstance(image, PIL.Image.Image):
        img = keras_image.load_img(image, target_size=image_size)
    else:
        img = image.resize(image_size)

    if img.mode != 'RGB':
        img = img.convert('RGB')

    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    probabilities = model.predict(img_array)
    class_index = np.argmax(probabilities)
    print(f"Best guess: {class_names[class_index]}")
    return class_names[class_index]

# Other helper functions
# ...

def main():
    buoy_list_df = pd.read_csv("scripts/working_buoys.csv")
    skip_buoy_list = pd.read_csv("scripts/failing_buoys.csv")["station_id"].tolist()
    buoy_urls = buoy_list_df["station_id"].tolist()
    base_output_path = "scripts/output"

    model = load_model(MODEL_PATH)

    for buoy_url in tqdm(buoy_urls, desc="Processing buoys"):
        print(f'Processing {buoy_url}', end='', flush=True)
        buoy_output_df, processed_images, processed_panels, processed_aligned, processed_stitched, processed_classifications, processed_mse, processed_horizon = process_buoy(buoy_url, model, base_output_path)

        if buoy_url in skip_buoy_list:
            print("Skipping")
            continue

if __name__ == "__main__":
    main()
