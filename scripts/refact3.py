import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import PIL
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image

# Import the functions from phase_one module
from phase_one import (download_image, resize_image_to_standard_height, split_image_into_panels, detect_horizon_line, align_horizon_line, mse, check_unusual_panels, stitch_aligned_images, load_skipped_buoys)

MODEL_PATH = 'models/gen3_keras/keras_model.h5'
IMAGE_SIZE = (224, 224)
CLASS_NAMES = ['Sunset', 'Storms', 'Normal', 'Object In View']
output_dir = "images"
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

def get_image_size(img):
    return img.size

def mse_between_arrays(arr1, arr2):
    try:
        return np.mean((arr1 - arr2) ** 2)
    except:
        return 0

def crop_the_bottom_off(image, filename):
    try:
        img = cv2.imread(image)
        height, width, channels = img.shape
        crop_img = img[0:height-100, 0:width]
        cv2.imwrite(filename, crop_img)
        return filename
    except:
        return image

def main():
    # Load the model
    model = load_model(MODEL_PATH)

    # Download the images
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Process the images
    for index, row in tqdm(buoy_list.iterrows(), total=buoy_list.shape[0]):
        # Download the image
        image_path = download_image(row['image_url'], output_dir)
        image_path = crop_the_bottom_off(image_path, image_path)
        # Resize the image
        resized_image_path = resize_image_to_standard_height(image_path, output_dir)
        # Split the image into panels
        image_panels = split_image_into_panels(resized_image_path, output_dir)
        # Detect the horizon line
        horizon_line = detect_horizon_line(image_panels, output_dir)
        # Align the horizon line
        aligned_panels = align_horizon_line(image_panels, horizon_line, output_dir)
        # Check for unusual panels
        unusual_panels = check_unusual_panels(aligned_panels, output_dir)
        # Stitch the aligned panels
        stitched_image = stitch_aligned_images(aligned_panels, unusual_panels, output_dir)
        # Classify the image
        image_class = classify_image(stitched_image, model)
        # Save the image
        output_path = os.path.join(output_dir, f"{row['id']}.jpg")
        stitched_image.save(output_path)
        # Save the image class
        buoy_list.loc[index, 'image_class'] = image_class
        # Save the image size
        buoy_list.loc[index, 'image_size'] = get_image_size(stitched_image)
        # Save the MSE
        buoy_list.loc[index, 'mse'] = mse_between_arrays(np.array(stitched_image), np.array(image_panels[0]))
        # Save the horizon line
        buoy_list.loc[index, 'horizon_line'] = horizon_line
        # Save the unusual panels
        buoy_list.loc[index, 'unusual_panels'] = unusual_panels


if __name__ == "__main__":
    main()
