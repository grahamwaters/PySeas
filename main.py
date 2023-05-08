import os
import time
import requests
from PIL import Image
from io import BytesIO
from datetime import datetime
from image_classifier import ImageClassifier
import pandas as pd
from tensorflow.keras.models import load_model
from ratelimit import limits, sleep_and_retry
import random


# load the model from the model folder
# MODEL_PATH = 'models/gen3_keras/keras_model.h5'
MODEL_PATH = 'models/improved_model/keras_model.h5'
BLANK_OR_NOT_MODEL_PATH = 'models/blank_or_not_model/keras_model.h5'
white_mode = True # set to True to use the blank_or_not_model to determine if the image is blank or not
only_save_originals = True # set to True to only save the original images and skip saving the panels individually.
white_model = load_model(BLANK_OR_NOT_MODEL_PATH)
model = load_model(MODEL_PATH)
CLASS_NAMES = [
    "sunset",
    "night",
    "white",
    "storm",
    "clouds",
    "strange sky",
    "object",
    "normal"
]


# Function to split and trim images
def process_image(image):
    """
    The process_image function takes an image and returns a list of 6 images, representing the 6 horizonally segmented panels of the original image.
    The function crops the original image into six equal parts, then removes 30 pixels from the bottom to remove
    the date/time stamp.

    :param image: Pass in the image that will be processed
    :return: A list of 6 cropped images
    :doc-author: Trelent
    """

    # Get the size of the image
    width, height = image.size
    # set the width of the panels to 1/6 of the total width
    panel_width = width / 6
    # set the height of the panels to the total height
    panel_height = height
    # set the starting x coordinate to 0
    x1 = 0
    # set the starting y coordinate to 0
    y1 = 0
    # set the ending x coordinate to the width of the panel
    x2 = panel_width
    # set the ending y coordinate to the height of the panel
    y2 = panel_height

    # Create a list to store the panels
    panels = []
    # Loop through the total number of panels (6)
    for i in range(6):
        # Crop the image
        panel = image.crop((x1, y1, x2, y2))
        # Add the panel to the list
        panels.append(panel)
        # Update the x coordinates to start where the previous panel ended
        x1 += panel_width
        x2 += panel_width
    # set temp2.png to be the original image
    image.save('temp2.png') # save the original image
    # Create a list to store the trimmed panels
    trimmed_panels = []
    # Loop through the panels
    for panel in panels:
        # Get the size of the panel
        width, height = panel.size
        # Crop 30 pixels from the bottom of the panel
        panel = panel.crop((0, 0, width, height - 30))
        #? upscale the image by 4x
        # panel = panel.resize((panel.width * 4, panel.height * 4))
        # # increase the resolution of the image
        # panel = panel.resize((panel.width * 4, panel.height * 4))
        # Add the trimmed panel to the list
        trimmed_panels.append(panel)
        # update temp2.png to be the trimmed panel
    # Return the trimmed panels
    return trimmed_panels

from tqdm import tqdm
# Function to download, process, and classify images
def analyze_buoys(model, blank_or_not_model=white_model):
    """
    The analyze_buoys function takes a model and an optional blank_or_not_model as arguments.
    It then iterates through the buoy urls, downloading each image and processing it into panels.
    The function classifies each panel using the ImageClassifier class, which uses the provided model to classify images.
    If any of those panels are classified as &quot;White&quot;, that buoy is skipped (removed from buoy_urls) and added to skip_buoy list.

    :param model: Specify which model to use for the classification
    :param blank_or_not_model: Determine if the image is blank or not
    :return: A list of the buoys that were skipped due to white panels
    :doc-author: Trelent
    """

    classifier = ImageClassifier()
    global white_mode
    global only_save_originals
    global buoy_urls

    for url in tqdm(buoy_urls):
        response = requests.get(url)
        if response.status_code != 200:
            print(f'Error downloading {url}')
            with open('scripts/failing_buoys.csv', 'a') as f:
                f.write(f'{url}\n')
                f.write(f'https://www.ndbc.noaa.gov/buoycam.php?station={url.split("/")[-1].split("=")[-1]}\n')
            continue
        image = Image.open(BytesIO(response.content))

        panels = process_image(image)
        classifications = [classifier.classify_image(panel, model, blank_or_not_model) for panel in panels]

        # check for any panels that are "White"
        # remove those buoys from the list
        if white_mode:
            if any(c == "White" for c in classifications):
                print(f'White panel found in {url}')
                skip_buoy_list.append(url)
                buoy_urls.remove(url)
                # extract the buoy id from the url
                buoy_id = url.split('/')[-1].split('=')[-1]
                # add the buoy to the failing_buoys.csv file
                with open('scripts/failing_buoys.csv', 'a') as f:
                    f.write(f'{buoy_id}\n')
                    f.write(f'https://www.ndbc.noaa.gov/buoycam.php?station={buoy_id}\n')
                # continue to the next buoy in the list
                continue

        # Check if any panel is not "normal"
        if any(c != "normal" for c in classifications):
            # get the most common classification
            c = max(set(classifications), key=classifications.count)
            # if the max is storm and there is a "sunset" panel then set the classification to "sunset"
            # if the max is normal and there is a "sunset" panel then set the classification to "sunset"
            # if the max is normal and there is a "night" panel then set the classification to "night"
            if c == "storm" and "sunset" in classifications:
                c = "sunset"
            elif c == "normal" and "sunset" in classifications:
                c = "sunset"
            elif c == "normal" and "night" in classifications:
                c = "night"

            c = c.lower().replace(" ", "_")

            # get the index of the classification
            i = classifications.index(c)
            # get the panel that is not "normal"
            panel = panels[i]
            # if any are not normal save ALL of the panels to the folder
            panel_set = panels
            # save the panel to a folder named by the classification and timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            #!folder_name = f"{c}/{url.split('/')[-2]}_{timestamp}"
            folder_name = str(c)
            os.makedirs(folder_name, exist_ok=True)
            # save all of the panels to the folder from the panel_set
            if not only_save_originals:
                for ii, pan in enumerate(panel_set):
                    # save the panel in Storms > {buoy_id} > {classification}_{timestamp}.jpg format
                    pan.save(os.path.join(folder_name, f"{ii}_[{c}]_{timestamp}.jpg"))
            # save the original noncropped image to the folder as well
            filename = f"{url.split('/')[-2]}_{timestamp}.jpg" # get the filename, which should be the buoy id and timestamp
            filename = filename.replace(" ", "_") # replace any spaces with underscores
            # if the image is not already in the folder, save it
            # save the panel as Storms / {buoy_id}_{classification}_{timestamp}.jpg
            # if it already exists, don't save it
            ext_filename = os.path.join(folder_name, filename).replace('/',"_") # replace any slashes with underscores
            if not os.path.exists(ext_filename):
                image.save(os.path.join(folder_name, ext_filename))# save the original image to the folder as well
            # # remove the buoy from the list
            # buoy_urls.remove(url)
            # # add the buoy to the failing_buoys.csv file
            # with open('scripts/failing_buoys.csv', 'a') as f:
            #     f.write(f'{url}\n')
            # # print the url and classification  to the console

# Main loop
while True:
    # Define buoy URLs from the working_buoys.csv file in the scripts folder
    buoy_urls = pd.read_csv("scripts/working_buoys.csv")["station_id"].tolist()
    # remove any buoys that are in the failing_buoys.csv file in the scripts folder
    skip_buoy_list = pd.read_csv("scripts/failing_buoys.csv")["station_id"].tolist()

    buoy_urls = [f'https://www.ndbc.noaa.gov/buoycam.php?station={station}' for station in buoy_urls]
    # remove duplicates from skip_buoy_list
    skip_buoy_list = list(dict.fromkeys(skip_buoy_list))
    # make it a list of urls again instead of station_ids
    skip_buoy_list = [f'https://www.ndbc.noaa.gov/buoycam.php?station={station}' for station in skip_buoy_list]
    buoy_urls = [url for url in buoy_urls if url not in skip_buoy_list]

    print(f'Number of buoys to process: {len(buoy_urls)}')
    print(f'Eliminated {len(skip_buoy_list)} buoys')
    random.shuffle(buoy_urls)
    analyze_buoys(model)
    print(f'Waiting 10 minutes')
    time.sleep(600)  # Wait for 10 minutes
    # random sample 10 of the white buoys and add them back to the list
    if white_mode:
        buoy_urls = random.sample(skip_buoy_list, 10) + buoy_urls
        skip_buoy_list = []
        # white_mode = False
    # if there are no buoys left, exit the program
    if len(buoy_urls) == 0:
        print('No buoys left to process')
        break
