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
import PIL
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image

# Import the functions from phase_one module
from phase_one import (download_image, resize_image_to_standard_height, split_image_into_panels, detect_horizon_line, align_horizon_line, mse, check_unusual_panels, stitch_aligned_images, load_skipped_buoys)

MODEL_PATH = 'models/gen3_keras/keras_model.h5'
IMAGE_SIZE = (224, 224)
CLASS_NAMES = ['Sunset', 'Storms', 'Normal', 'Object In View']
output_dir = "src/output"
collecting_all = True
buoy_list = pd.read_csv("src/working_buoys.csv")

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


def get_image_size(img):
    return img.size

def mse_between_arrays(arr1, arr2):
    try:
        return np.mean((arr1 - arr2) ** 2)
    except:
        return 0

def crop_the_bottom_off(image, filename):
    try:
        img_width, img_height = image.size
        cropped_image = image.crop((0, 0, img_width, img_height - 20))

        # Use the original format when saving the cropped image
        file_format = image.format

        cropped_image.save(f"cropped_{filename}", format=file_format)
        return cropped_image
    except Exception as e:
        print("Error cropping the bottom off of the image: " + str(e))
        return image  # Return the original image if there's an error

def process_buoy(buoy_url, model, base_output_path):
    buoy_id = buoy_url.split("/")[-1]
    buoy_id = buoy_id.split(".")[0]

    buoy_output_path = os.path.join(base_output_path, buoy_id)
    if not os.path.exists(buoy_output_path):
        os.mkdir(buoy_output_path)

    buoy_images_path = os.path.join(buoy_output_path, "images")
    if not os.path.exists(buoy_images_path):
        os.mkdir(buoy_images_path)

    buoy_panels_path = os.path.join(buoy_output_path, "panels")
    if not os.path.exists(buoy_panels_path):
        os.mkdir(buoy_panels_path)

    buoy_aligned_path = os.path.join(buoy_output_path, "aligned")
    if not os.path.exists(buoy_aligned_path):
        os.mkdir(buoy_aligned_path)

    buoy_stitched_path = os.path.join(buoy_output_path, "stitched")
    if not os.path.exists(buoy_stitched_path):
        os.mkdir(buoy_stitched_path)

    buoy_output_csv = os.path.join(buoy_output_path, "output.csv")

    # Load the list of buoys that we've already processed
    # if it does not exist, create it
    if not os.path.exists(buoy_output_csv):
        with open(buoy_output_csv, "w") as f:
            f.write("image,panel,aligned,stitched,classification,mse,horizon\n")
    buoy_output_df = pd.read_csv(buoy_output_csv)

    # Get the list of images that we've already processed
    processed_images = buoy_output_df["image"].tolist()

    # Get the list of images that we've already processed
    processed_panels = buoy_output_df["panel"].tolist()

    # Get the list of images that we've already processed
    processed_aligned = buoy_output_df["aligned"].tolist()

    # Get the list of images that we've already processed
    processed_stitched = buoy_output_df["stitched"].tolist()

    # Get the list of images that we've already processed
    processed_classifications = buoy_output_df["classification"].tolist()

    # Get the list of images that we've already processed
    processed_mse = buoy_output_df["mse"].tolist()

    # Get the list of images that we've already processed
    processed_horizon = buoy_output_df["horizon"].tolist()

    # Get the list of images that we've already processed
    # processed_panels_unusual = buoy_output_df["panels_unusual"].tolist()

    # Get the list of images that we've already processed
    # processed_panels_unusual_reason = buoy_output_df["panels_unusual_reason"].tolist()
    return buoy_output_df, processed_images, processed_panels, processed_aligned, processed_stitched, processed_classifications, processed_mse, processed_horizon

def main():
    buoy_list_df = pd.read_csv("src/working_buoys.csv")
    skip_buoy_list = pd.read_csv("src/failing_buoys.csv")["station_id"].tolist()
    buoy_urls = buoy_list_df["station_id"].tolist()
    base_output_path = "src/output"
    # The rest of the main function body remains the same

    model = load_model(MODEL_PATH)

    for buoy_url in tqdm(buoy_urls, desc="Processing buoys"):
        print(f'Processing {buoy_url}', end='', flush=True)
        buoy_output_df, processed_images, processed_panels, processed_aligned, processed_stitched, processed_classifications, processed_mse, processed_horizon = process_buoy(buoy_url, model, base_output_path)

        if buoy_url in skip_buoy_list:
            print("Skipping")
            continue

if __name__ == "__main__":
    main()
