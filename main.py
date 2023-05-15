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
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from colorama import Fore, Back, Style

# import ImageDraw module
import textwrap
from PIL import ImageFont, ImageDraw

# load the model from the model folder
# MODEL_PATH = 'models/gen3_keras/keras_model.h5'
# MODEL_PATH = 'models/latest_model/keras_model.h5'
MODEL_PATH = 'models/converted_keras/keras_model.h5'
BLANK_OR_NOT_MODEL_PATH = 'models/blank_or_not_model/keras_model.h5'

#! Flags
white_mode = False # set to True to use the blank_or_not_model to determine if the image is blank or not
only_save_originals = True # set to True to only save the original images and skip saving the panels individually.
save_confidence_plots = False # set to True to save the confidence plots

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
    "normal",
    "moon"
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

        # if the image is not mostly white then save it
        avg_color = np.array(image).mean(axis=(0,1))
        if avg_color.mean() > 300:
            # print(f'White image found in {url}')
            # skip_buoy_list.append(url)
            # buoy_urls.remove(url)
            # # save the image as 'image_white.png'
            # image.save('image_white.png')
            continue
        else:
            # save it regardless of the color as the url + timestamp + .png
            image.save(f'raw/{url.split("/")[-1].split("=")[-1]}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        panels = process_image(image)
        #!classifications = [classifier.classify_image(panel, model, blank_or_not_model) for panel in panels]
        classifications = []
        # save each panel with the pattern cropped_panel_#.png where # is the index of the panel, annotated with the classification of the panel


        for panel in panels:
            if white_mode:
                # check if the majority of pixels are white
                avg_color = np.array(panel).mean(axis=(0,1))
                if avg_color.mean() > 250:
                    classification = 'White'
                if avg_color.mean() < 10:
                    classification = 'night'
                if avg_color.max() - avg_color.min() > 10: # the spread is indicative of a nonblank image
                    classification = classifier.classify_image(panel, model)
                else:
                    classification = 'normal'
                if avg_color.mean() < 250 and classification == 'White':
                    classification = 'normal'
                panel.save('this_panel.png')
            else:
                panel.save('this_panel.png')
                classification = classifier.classify_image(panel, model)
            panel_avg_color = np.array(panel).mean(axis=(0,1)) # get the average color of the panel
            # if the panel is not white then save it
            if panel_avg_color.mean() < 250:
                # classify the panel
                classification = classifier.classify_image(panel, model)
                # save the panel with the classification in the filename
                panel.save(f'panels/{url.split("/")[-1].split("=")[-1]}_{classification}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
            print(f'Classification: {classification} for {url} because of {panel} with average color of {avg_color.mean()}')
            classifications.append(classification)
            if classification != "White":
                classifications.append(classification)
                if classification == 'storm':
                    print(Fore.RED + f'Storm panel found in {url}')
                elif classification == 'normal':
                    print(Fore.GREEN + f'Clear panel found in {url}')
                elif classification == 'clouds':
                    print(Fore.YELLOW + f'Cloudy panel found in {url}')
                elif classification == 'night':
                    print(Fore.BLUE + f'Night panel found in {url}')
                elif classification == 'strange sky':
                    print(Fore.MAGENTA + f'Strange sky panel found in {url}')
                elif classification == 'unknown':
                    print(Fore.CYAN + f'Unknown panel found in {url}')
                else:
                    print(Fore.WHITE + f'White panel found in {url}')
            else:
                continue
        for i, panel in enumerate(panels):
            panel_annotated = panel.copy()
            draw = ImageDraw.Draw(panel_annotated)
            # make the text white, and large
            draw.text((0, 0), classifications[i], (255, 255, 255))
            panel_annotated.save(f'cropped_panel_{i}.png')
        # check for any panels that are "White"
        # remove those buoys from the list
        if white_mode:
            if all(c=="White" for c in classifications) and classifications != []: #was any
                continue
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
            else:
                print(f'White panel not found in {url}')
        # Check if any panel is not "normal"
        if any(c != "normal" for c in classifications) and classifications != [] and \
            all(c != "White" for c in classifications):
            # get the most common classification
            c = max(set(classifications), key=classifications.count)
            x = "object"
            y = "white"
            all_white = all(y == x for x in classifications)
            any_c = any(c == x for x in classifications)
            # if the max is storm and there is a "sunset" panel then set the classification to "sunset"
            # if the max is normal and there is a "sunset" panel then set the classification to "sunset"
            # if the max is normal and there is a "night" panel then set the classification to "night"
            if c == "storm" and "sunset" in classifications:
                c = "sunset"
            elif c == "normal" and "sunset" in classifications:
                c = "sunset"
            elif c == "normal" and "night" in classifications:
                c = "night"
            # if a panel was classified with 'object' then set the classification to 'object'
            elif c == "object":
                c = "object"
            # if a panel was classified with 'white' then skip the buoy
            elif all_white:
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
            # c = c.lower().replace(" ", "_")

            # get the index of the classification
            i = classifications.index(c)
            # get the panel that is not "normal"
            panel = panels[i]
            # get the timestamp of the panel
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            # save the panel that was not normal at 6x resolution
            # annotate the panel with the classification and confidence score
            if save_confidence_plots:
                fig, ax = plt.subplots()
                ax.imshow(panel)
                text = f'{c} {np.max(classifications.count(c)) / len(classifications):.2f}'
                ax.annotate(text, (0, 0), (0, -20), xycoords='axes fraction', textcoords='offset points', va='top')
                ax.axis('off')
                ax.set_aspect('equal')
                fig.tight_layout()
                fig.canvas.draw()
                # convert the figure to a numpy array
                panel = np.array(fig.canvas.renderer._renderer)
                # close the figure
                plt.close(fig)
                # upscale the panel to 6x
                try:
                    panel = Image.fromarray(panel).resize((panel.width * 6, panel.height * 6), Image.BICUBIC)
                except AttributeError:
                    # numpy.ndarray has no attribute 'width'
                    panel = Image.fromarray(panel).resize((panel.shape[1] * 6, panel.shape[0] * 6), Image.BICUBIC)
                    # to avoid OSError: cannot write mode RGBA as JPEG
                    panel = panel.convert('RGB')

                # save the panel to the folder named by the classification and timestamp
                panel.save(f"{url.split('/')[-2]}_{timestamp}_6x.jpg")

            # if any are not normal save ALL of the panels to the folder
            panel_set = panels
            # save the panel to a folder named by the classification and timestamp

            #!folder_name = f"{c}/{url.split('/')[-2]}_{timestamp}"
            folder_name = str(c)
            os.makedirs(folder_name, exist_ok=True)
            # save all of the panels to the folder from the panel_set
            if not only_save_originals:
                for ii, pan in enumerate(panel_set):
                    # save the panel in Storms > {buoy_id} > {classification}_{timestamp}.jpg format
                    pan.save(os.path.join(folder_name, f"{ii}_[{c}]_{timestamp}.jpg"))


            # upscale the original image to 4x
            image = image.resize((image.width * 4, image.height * 4), Image.BICUBIC)
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

def refine_list(buoy_urls, skip_buoy_list):
    # using the prefixes of the files in the /raw folder, build a list of non-malfunctioing buoys
    # get the list of files in the /raw folder
    files = os.listdir('raw')
    # get the list of buoys from the files
    buoys = [file.split('_')[0] for file in files]
    # remove duplicates
    buoys = list(dict.fromkeys(buoys))
    return buoys


first_buoys = pd.read_csv("scripts/manual_buoys.csv")["station_id"].tolist()
# refined_buoys = if len(refine_list(buoy_urls, skip_buoy_list)) > 0: refine_list(buoy_urls, skip_buoy_list) else: first_buoys
skip_buoy_list = []
# Define buoy URLs from the working_buoys.csv file in the scripts folder
first_buoys = pd.read_csv("scripts/manual_buoys.csv")["station_id"].tolist()
buoy_urls = pd.read_csv("scripts/working_buoys.csv")["station_id"].tolist()
buoy_urls = first_buoys + buoy_urls
# get unique values
buoy_urls = list(dict.fromkeys(buoy_urls))
if len(refine_list(buoy_urls, skip_buoy_list)) > 0:
    refined_buoys = refine_list(buoy_urls, skip_buoy_list)
else:
    refined_buoys = first_buoys
# Main loop
loop_count = 0
same = 0
while True:

    #! IF - we have looped at least 5 times and the length of unique buoys has not changed (i.e. refined_buoys length has not changed) then stop reading the file and use the refined_buoys list with no dupes for the list of buoys to scrape
    last_length_refined = len(refined_buoys)
    loop_count += 1
    if loop_count > 5 and len(refined_buoys) == last_length_refined:
        first_buoys = refined_buoys # this is the list of buoys that will be used to scrape now make them urls
        buoy_urls = [f'https://www.ndbc.noaa.gov/buoycam.php?station={station}' for station in first_buoys]
        print(Fore.GREEN + f'Using refined_buoys list with {len(refined_buoys)} buoys')
        print(Fore.GREEN + f'Loop count: {loop_count}', Style.RESET_ALL)
    else: # otherwise, keep reading the file and adding buoys to the list
        # Define buoy URLs from the working_buoys.csv file in the scripts folder
        first_buoys = pd.read_csv("scripts/manual_buoys.csv")["station_id"].tolist()
        buoy_urls = pd.read_csv("scripts/working_buoys.csv")["station_id"].tolist()
        buoy_urls = first_buoys + buoy_urls
        # get unique values
        buoy_urls = list(dict.fromkeys(buoy_urls))
        # remove any buoys that are in the failing_buoys.csv file in the scripts folder
        skip_buoy_list = pd.read_csv("scripts/failing_buoys.csv")["station_id"].tolist()
        # remove duplicates from skip_buoy_list
        skip_buoy_list = list(dict.fromkeys(skip_buoy_list))
        # make it a list of urls again instead of station_ids
        skip_buoy_list = [f'https://www.ndbc.noaa.gov/buoycam.php?station={station}' for station in skip_buoy_list]
        # remove any buoys that are in the skip_buoy_list
        buoy_urls = [buoy for buoy in buoy_urls if buoy not in skip_buoy_list]
        # if len(refine_list(buoy_urls, skip_buoy_list)) > 0:
        #     refined_buoys = refine_list(buoy_urls, skip_buoy_list)
        # else:
        #     refined_buoys = first_buoys
        refined_buoy_urls = [f'https://www.ndbc.noaa.gov/buoycam.php?station={station}' for station in refined_buoys]
        print(f"refined_buoys: {refined_buoys}")
        print(f"buoy_urls: {buoy_urls}")
        print(f"skip_buoy_list: {skip_buoy_list}")
        print(f"first_buoys: {first_buoys}")
        print(f"loop_count: {loop_count}")
        print(f"same: {same}")
        print(f"last_length_refined: {last_length_refined}")
        print(f"len(refined_buoys): {len(refined_buoys)}")
        print(f"len(buoy_urls): {len(buoy_urls)}")
        print(f"len(skip_buoy_list): {len(skip_buoy_list)}")
        print(f"len(first_buoys): {len(first_buoys)}")
        print(f"len(buoy_urls): {len(buoy_urls)}")
        print(f"len(skip_buoy_list): {len(skip_buoy_list)}")
        print(f"len(first_buoys): {len(first_buoys)}")
        print(f"len(buoy_urls): {len(buoy_urls)}")
        print(f"len(skip_buoy_list): {len(skip_buoy_list)}")
        buoy_urls = refined_buoy_urls
    print(f'Number of buoys to process: {len(buoy_urls)}')
    print(f'Eliminated {len(skip_buoy_list)} buoys')
    #random.shuffle(buoy_urls)
    analyze_buoys(model)
    print(f'Waiting 10 minutes')
    time.sleep(300)  # Wait for 10 minutes
    # random sample 30 of the white buoys and add them back to the list
    # remove dupes from first_buoys
    first_buoys = list(dict.fromkeys(first_buoys))
    if white_mode:
        buoy_urls = random.sample(skip_buoy_list, 50) + buoy_urls + first_buoys
        skip_buoy_list = []
        # white_mode = False
    # if there are no buoys left, exit the program
    if len(buoy_urls) == 0:
        print('No buoys left to process')
        break
