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

collecting_all = True
buoy_list = pd.read_csv("src/working_buoys.csv")

import numpy as np
from PIL import Image

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
import PIL
# MODEL_PATH = 'models/buoy_model/keras_model.h5'
MODEL_PATH = 'models/gen3_keras/keras_model.h5'
IMAGE_SIZE = (224, 224)
# CLASS_NAMES = ['Direct Sun', 'Stormy Weather', 'Interesting', 'Object Detected', 'Sunset', 'Clouds', 'Night']
CLASS_NAMES = ['Sunset', 'Storms', 'Normal', 'Object In View']
# 0 Sunset
# 1 Storms
# 2 Normal
# 3 Object In View

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
    # print(f"Best guess: {class_names[class_index]}")
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





@sleep_and_retry
@limits(calls=15, period=60)
def download_image(image_url):
    response = requests.get(image_url)
    if response.status_code != 200:
        print(f"API response: {response.status_code}")
        return None

    img = Image.open(BytesIO(response.content)).convert('L')  # Convert image to grayscale
    min_value, max_value = img.getextrema()
    if max_value < 200:
        # Rest of the code
        print(f"\nImage too white: {image_url}")

        filename = "src/failing_buoys.csv"
        if os.path.isfile(filename):
            with open(filename, "r") as f:
                failing_buoys = set(f.read().splitlines())
        else:
            failing_buoys = set()

        if image_url not in failing_buoys:
            with open(filename, "a") as f:
                f.write(image_url + "\n")

        buoy_id = image_url.split("/")[-1].split(".")[0]

        if buoy_id not in failing_buoys:
            with open(filename, "a") as f:
                f.write(buoy_id + "\n")

        return None
    else:
        # the image is not too white
        # save it as temp.png in the current directory

        img.save("temp.png")
    return img

def resize_image_to_standard_height(image, target_height):
    if image is None:
        return None
    width, height = image.size
    new_height = target_height
    new_width = int((new_height / height) * width)
    return image.resize((new_width, new_height), Image.ANTIALIAS)

def split_image_into_panels(resized_image, num_panels):
    if resized_image is None:
        return None
    width, height = resized_image.size
    panel_width = width // num_panels

    panels = []
    for i in range(num_panels):
        left = i * panel_width
        right = left + panel_width
        panel = resized_image.crop((left, 0, right, height))
        panel = panel.crop((0, 0, panel_width, height - 38))
        panels.append(panel)

    return panels

# def check_unusual_panels(panels, mse_threshold):
#     unusual_panels = []
#     rich_color_panels = []
#     panel_mses = []
#     for panel in panels:
#         panel_array = np.array(panel)
#         panel_mse = mse_between_arrays(panel_array, panel_array)
#         panel_mses.append(panel_mse)

#     median_mse = np.median(panel_mses)
#     rich_color_threshold = median_mse * 0.8

#     for i, panel in enumerate(panels):
#         panel_array = np.array(panel)
#         panel_mse = panel_mses[i]
#         if panel_mse > mse_threshold:
#             unusual_panels.append(i)
#         if panel_mse > rich_color_threshold:
#             rich_color_panels.append(panel)
#     return unusual_panels, rich_color_panels


def detect_horizon_line(img):
    img = np.array(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Use a Gaussian blur to reduce noise before edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

    # Increase the threshold for HoughLinesP to reduce false positives
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 150, minLineLength=100, maxLineGap=20)

    if lines is None:
        return 0

    try:
        # Select the line closest to the horizontal orientation
        closest_horizontal_line = min(lines, key=lambda line: abs(np.arctan2(line[0][3] - line[0][1], line[0][2] - line[0][0])))
        x1, y1, x2, y2 = closest_horizontal_line[0]
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        return angle
    except Exception as e:
        print(f"Error: {e}")
        return 0

def align_horizon_line(img):
    img = np.array(img)
    original = img.copy()
    tilt_angle = detect_horizon_line(img)
    height, width = img.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, tilt_angle, 1)
    aligned_img = cv2.warpAffine(img, rotation_matrix, (width, height), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)

    fixed = aligned_img.copy()

    fig, axs = plt.subplots(1, 2, figsize=(20, 20))
    axs[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axs[0].set_title("Original")
    axs[1].imshow(cv2.cvtColor(fixed, cv2.COLOR_BGR2RGB))
    axs[1].set_title("Aligned {}".format(tilt_angle))
    plt.savefig('latest_alignment.png')

    return aligned_img

def mse(image_a, image_b):
    try:
        return np.mean((image_a - image_b) ** 2)
    except:
        return 0

def check_unusual_panels(panels, mse_threshold=1):
    """
    The check_unusual_panels function takes in a list of panels and returns the indices of the unusual panels.

    :param panels: Pass the list of panels to be checked
    :param mse_threshold: Determine the threshold for which a panel is considered unusual
    :return: A list of panel indices that are unusual, a list of panel indices that have rich colors, and a boolean indicating whether the image is too white
    :doc-author: Trelent
    """

    unusual_panels = []
    rich_color_panels = []
    image_too_white = False

    for i, panel in enumerate(panels):
        mse_value = mse(panel, np.full_like(panel, 255))
        if mse_value < mse_threshold:
            unusual_panels.append(i)
        else:
            rich_color_panels.append(i)

        if mse(panel, np.full_like(panel, 255)) < 5:
            image_too_white = True
        # save the panel image to `temp.png`
        try:
           panel.save('temp.png')
           if not image_too_white:
            time.sleep(1)
            panel.save('temp2.png')
            return unusual_panels, rich_color_panels, image_too_white
        except:
            return unusual_panels, rich_color_panels, image_too_white
    return panels, panels, False

def detect_horizon_line(img):
    if isinstance(img, PIL.Image.Image):
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    if len(img.shape) == 2 or img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # The rest of the function

    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)

    try:
        longest_line = max(lines, key=lambda line: np.linalg.norm(line[0][2:] - line[0][:2]))
        x1, y1, x2, y2 = longest_line[0]
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        return angle
    except TypeError:
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 0

def align_horizon_line(img):
    img = np.array(img)
    original = img.copy()
    tilt_angle = detect_horizon_line(img)
    height, width = img.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, tilt_angle, 1)
    aligned_img = cv2.warpAffine(img, rotation_matrix, (width, height), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)

    fixed = aligned_img.copy()

    fig, axs = plt.subplots(1, 2, figsize=(20, 20))
    axs[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axs[0].set_title("Original")
    axs[1].imshow(cv2.cvtColor(fixed, cv2.COLOR_BGR2RGB))
    axs[1].set_title("Aligned {}".format(tilt_angle))
    plt.savefig('latest_alignment.png')

    return aligned_img

def stitch_aligned_images(aligned_images):
    stitcher = cv2.createStitcher() if imutils.is_cv3() else cv2.Stitcher_create()
    if len(aligned_images) < 2:
        print("Not enough images to stitch")
        return None
    orb = cv2.ORB_create(nfeatures=1000)

    for i, img in enumerate(aligned_images):
        keypoints, _ = orb.detectAndCompute(img, None)
        print(f"Image {i + 1} has {len(keypoints)} features")

    (status, stitched_image) = stitcher.stitch(aligned_images)

    if status == 0:
        return stitched_image
    else:
        return None

import random
from tqdm import tqdm
# import shuffle

def load_skipped_buoys(filename):
    with open(filename, 'r') as f:
        skipped_buoys = [line.strip() for line in f.readlines()]
    return skipped_buoys


def main():

    buoy_list_df = pd.read_csv("src/working_buoys.csv")
    skip_buoy_list = pd.read_csv("src/failing_buoys.csv")["station_id"].tolist()
    # make unique list of buoys to process
    # remove dupes from skip_buoy_list
    skip_buoy_list = list(set(skip_buoy_list))
    buoy_list = buoy_list_df["station_id"].tolist()
    # remove any skipped buoys from the list
    buoy_list = [buoy for buoy in buoy_list if buoy not in skip_buoy_list]
    print(f'There are {len(buoy_list)} buoys to process, we have identified {len(skip_buoy_list)} buoys to skip')
    # skipped_buoys_filename = 'src/failing_buoys.csv'
    # skipped_buoys = load_skipped_buoys(skipped_buoys_filename)



    random.shuffle(buoy_list)

    buoy_urls = [f'https://www.ndbc.noaa.gov/buoycam.php?station={station}' for station in buoy_list]
    buoy_urls = [url for url in buoy_urls if url not in skip_buoy_list] # remove any skipped buoys from the list


    base_output_path = "images/output_test"

    print(f'There are {len(buoy_urls)} buoys to process')

    model = load_model(MODEL_PATH)

    for buoy_url in tqdm(buoy_urls, desc="Processing buoys"):

        print(f'Processing {buoy_url}', end='', flush=True)
        image = download_image(buoy_url)
        if image is None:
            continue

        resized_image = resize_image_to_standard_height(image, target_height=640)
        panels = split_image_into_panels(resized_image, num_panels=6)
        # crop the panels to remove the black border on the bottom of the image using crop_the_bottom_off function
        # Update the list comprehension in your main function
        # Update the list comprehension in your main function to pass the filename
        panels = [crop_the_bottom_off(panel, f"panel_{i}.png") for i, panel in enumerate(panels)]


        unusual_panels, rich_color_panels, image_too_white = check_unusual_panels(panels, mse_threshold=1)

        if image_too_white:
            print(" - Image too white, skipping")
            with open('failing_buoys.csv', 'a') as f:
                f.write(f'{buoy_url}\n')
            continue


        # Rest of the code


        resized_image = resize_image_to_standard_height(image, target_height=640)
        panels = split_image_into_panels(resized_image, num_panels=4)
        unusual_panels, rich_color_panels, image_too_white_bool = check_unusual_panels(panels, mse_threshold=1)
        if image_too_white_bool:
            # add the buoy to the list of failing buoys
            with open('failing_buoys.csv', 'a') as f:
                f.write(f'{buoy_url}\n')
            continue
        if unusual_panels:
            print(f" - Unusual panels: {unusual_panels}")
            current_datetime = datetime.now()
            date_str = current_datetime.strftime("%Y-%m-%d")
            time_str = current_datetime.strftime("%H-%M-%S")
            buoy_name = buoy_url.split("/")[-2]

            output_path = os.path.join(base_output_path, buoy_name, date_str, time_str)
            os.makedirs(output_path, exist_ok=True)

            for i, panel in enumerate(panels):
                # get the filepath for panel i
                path_i = os.path.join(output_path, f"panel_{i}.png")
                panel_classification = classify_image(panel, model)
                if panel_classification in ["Stormy Weather", "Sunset", "Interesting"]:
                    panel_output_path = os.path.join(output_path, f"panel_{i+1}.jpg")
                    print(f'Found {panel_classification} in panel {i+1}, saving to {panel_output_path}')
                    panel.save(panel_output_path)
                    # update temp.png by saving this panel to it
                    panel.save("temp.png")

            aligned_images = [align_horizon_line(panel) for panel in panels]
            stitched_image = stitch_aligned_images(aligned_images)

            panorama_output_path = os.path.join(output_path, "panorama.jpg")
            try:
                cv2.imwrite(panorama_output_path, stitched_image)

                latest_output_path = os.path.join(base_output_path, buoy_name, "latest.jpg")
                cv2.imwrite(latest_output_path, stitched_image)
            except AttributeError:
                print("Panorama could not be stitched")
            except Exception as e:
                pass




if __name__ == "__main__":
    main()
