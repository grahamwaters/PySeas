import os
import time
import shutil
import uuid
import random
import requests
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from io import BytesIO
from datetime import datetime
from tqdm import tqdm
from ratelimit import sleep_and_retry
from image_classifier import ImageClassifier
import pandas as pd
from tensorflow.keras.models import load_model

from PIL import ImageFont
import logging

import os
import math
from PIL import Image

if os.path.exists('log.txt'):
    os.remove('log.txt')
logging.basicConfig(filename='log.txt', level=logging.INFO, format='%(asctime)s %(message)s')

MODEL_PATH = 'models/buoy_model/keras_model.h5'

upscalefactor = 6
BLANK_OR_NOT_MODEL_PATH = 'models/blank_or_not_model/keras_model.h5'

white_mode = False
only_save_originals = True
save_confidence_plots = True
from colorama import Fore
model = load_model(MODEL_PATH)
white_model = load_model(BLANK_OR_NOT_MODEL_PATH)
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

if not os.path.exists('skip_buoy_list.txt'):
    with open('skip_buoy_list.txt', 'w') as f:
        f.write('')

def refine_list(buoy_urls, skip_buoy_list):
    """
    The refine_list function takes a list of buoys and removes any buoys that are not functioning.
    It does this by checking the /raw folder for files with the buoy number as a prefix. If there is no file, then it is assumed that the buoy is not working.

    :param buoy_urls: Pass the list of buoys to the function
    :param skip_buoy_list: Pass a list of buoys that are known to be malfunctioning
    :return: A list of buoys that are not in the skip_buoy_list
    :doc-author: Trelent
    """

    if not os.path.exists('raw'):
        os.mkdir('raw')
    files = os.listdir('raw')

    buoys = [file.split('_')[0] for file in files]

    buoys = list(dict.fromkeys(buoys))
    return buoys

@sleep_and_retry
def check_buoy_status(buoy_id):
    """
    The check_buoy_status function takes in a buoy_id and checks if the buoy is working.
    It does this by checking if more than 90% of the pixels are white. If they are then put a giant red word on the image that says &quot;NOT WORKING&quot;
    and save it as this_panel.png, otherwise just save it as this_panel.png

    :param buoy_id: Get the buoy_id from the user
    :return: True if the buoy is working and false if it's not
    :doc-author: Trelent
    """

    with open('skip_buoy_list.txt', 'r') as f:
        skip_buoy_list = f.read().splitlines()
    if buoy_id in skip_buoy_list:
        return False
    url = f'https://www.ndbc.noaa.gov/buoycam.php?station={buoy_id}'
    response = requests.get(url)
    print(f'Testing buoy {buoy_id}...')
    try:

        img = Image.open(BytesIO(response.content)).convert("L")

        total_pixels = img.width * img.height

        white_pixels = 0
        for pixel in img.getdata():
            if pixel > 200:
                white_pixels += 1

        if white_pixels / total_pixels > 0.9:

            draw = ImageDraw.Draw(img)

            font = ImageFont.truetype("Open_Sans/OpenSans-VariableFont_wdth,wght.ttf", 50)

            draw.text((img.width/2, img.height/2), f"Buoy {buoy_id} is NOT WORKING", fill='red', font=font, anchor='mm')

            img.save('this_panel.png')

            with open('skip_buoy_list.txt', 'a') as f:
                f.write(f'{buoy_id}\n')
            return False
        else:

            img.save('this_panel.png')

            img.save(f'raw/{buoy_id}_{random.randint(1,100000000)}_this_panel.png')
            return True

    except IOError:
        print("Error while reading image from", url)

        img.save('this_panel.png')
        return False
    except Exception as e:

        with open('log.txt', 'a') as f:
            f.write(f'Error while checking buoy {buoy_id}: {e}\n')
        return False

def process_image(image):
    """
    The process_image function takes an image and returns a list of 6 images, representing the 6 horizonally segmented panels of the original image.
    The function crops the original image into six equal parts, then removes 30 pixels from the bottom to remove
    the date/time stamp.

    :param image: Pass in the image that will be processed
    :return: A list of 6 cropped images
    :doc-author: Trelent
    """

    width, height = image.size

    panel_width = width / 6

    panel_height = height

    x1 = 0

    y1 = 0

    x2 = panel_width

    y2 = panel_height

    panels = []

    for i in range(6):

        panel = image.crop((x1, y1, x2, y2))

        panels.append(panel)

        x1 += panel_width
        x2 += panel_width

    image.save('temp2.png')

    trimmed_panels = []

    for panel in panels:

        width, height = panel.size

        panel = panel.crop((0, 0, width, height - 30))

        trimmed_panels.append(panel)

    return trimmed_panels

def classify_panel(panel, model, white_model):
    """
    The classify_panel function takes in a panel image and classifies it as one of the following:
        - 'white' if the average color is greater than 200 (i.e. white)
        - 'night' if the average color is less than 10 (i.e. black)
        - otherwise, it uses ImageClassifier to classify_image

    :param panel: Pass in the image of a panel
    :param model: Classify the panel as a day or night image
    :param white_model: Classify panels as white
    :return: The class of the panel
    :doc-author: Trelent
    """

    avg_color = np.array(panel).mean(axis=(0, 1))
    if avg_color.mean() > 200:
        return 'white'
    elif avg_color.mean() < 10:
        return 'night'
    else:
        return ImageClassifier.classify_image(panel, model, white_model)

def analyze_buoys(model, white_model, buoy_urls, save_confidence_plots=False, only_save_originals=False, white_mode=False):
    """
    The analyze_buoys function takes in a model and white_model, which are the models that will be used to classify
    the images. The function then iterates through each buoy url in the buoy_urls list, downloads the image from that url,
    and saves it as an Image object. It then processes this image into panels using process_image(), and classifies each panel
    using ImageClassifier().classify_image() with either model or white_model depending on whether or not it is a night time
    panel (determined by checking if its average color is less than 10). If there are 6 panels and their average

    :param model: Classify the image
    :param white_model: Classify the image as white or not
    :return: A list of images that are classified as stormy
    :doc-author: Trelent
    """

    classifier = ImageClassifier()

    buoy_urls = [str(buoy) for buoy in buoy_urls]

    print(f'Checking buoy status for {len(buoy_urls)} buoys...')
    buoy_urls = [buoy for buoy in buoy_urls if check_buoy_status(buoy)]
    print(f'{len(buoy_urls)} buoys are working.')

    for url in tqdm(buoy_urls):
        url = f'https://www.ndbc.noaa.gov/buoycam.php?station={url}'
        response = requests.get(url)
        if response.status_code != 200:
            print(f'Error downloading {url}')
            with open('scripts/failing_buoys.csv', 'a') as f:
                f.write(f'{url}\n')
                f.write(f'https://www.ndbc.noaa.gov/buoycam.php?station={url.split("/")[-1].split("=")[-1]}\n')
            continue
        image = Image.open(BytesIO(response.content))

        avg_color = np.array(image).mean(axis=(0,1))
        if avg_color.mean() > 300:
            continue
        else:

            image.save(f'raw/{url.split("/")[-1].split("=")[-1]}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        panels = process_image(image)

        classifications = [classifier.classify_image(panel, model, white_model) for panel in panels]

        panel_countedas_white = 0
        for panel in panels:
            white_mode = False
            if white_mode:

                avg_color = np.array(panel).mean(axis=(0,1))
                if avg_color.mean() > 250:
                    classification = 'white'
                if avg_color.mean() < 10:
                    classification = 'night'
                if avg_color.max() - avg_color.min() > 10:
                    classification = classifier.classify_image(panel, model)
                else:
                    classification = 'normal'
                if avg_color.mean() < 250 and classification == 'white':
                    classification = 'normal'
                panel.save('this_panel.png')
            else:
                panel.save('this_panel.png')
                classification = classifier.classify_image(panel, model)
            panel_avg_color = np.array(panel).mean(axis=(0,1))

            if len(panels) == 6 and panel_avg_color.mean() < 250:
                classification = classifier.classify_image(panel, model)

                if not os.path.exists(f'panels/{classification}'):
                    os.makedirs(f'panels/{classification}')
                panel.save(f'panels/{classification}/{url.split("/")[-1].split("=")[-1]}_{classification}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
            if panel_avg_color.mean() < 250:

                classification = classifier.classify_image(panel, model)

                panel.save(f'panels/{classification}/{url.split("/")[-1].split("=")[-1]}_{classification}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
            print(f'Classification: {classification} for {url} because of {panel} with average color of {avg_color.mean()}')
            classifications.append(classification)
            if classification != "white":
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
                    print(Fore.WHITE + f'white panel found in {url}')
            else:

                panel_countedas_white += 1
                if panel_countedas_white == 6:
                    print(Fore.WHITE + f'white panel found in {url}')
                    buoy_id = str(url.split('/')[-1].split('=')[-1])
                    try:
                        buoy_urls.remove(buoy_id)
                    except ValueError:
                        print(f"ValueError({buoy_id} not in list')")
                    with open('scripts/failing_buoys.csv', 'a') as f:
                        f.write(f'{buoy_id}\n')
                        f.write(f'https://www.ndbc.noaa.gov/buoycam.php?station={buoy_id}\n')

                continue

        for i, panel in enumerate(panels):
            panel_annotated = panel.copy()
            draw = ImageDraw.Draw(panel_annotated)

            draw.text((0, 0), classifications[i], fill='red')
            panel_annotated.save(f'cropped_panel_{i}.png')

        if white_mode:
            if all(c=="white" for c in classifications) and classifications != []:
                continue
                print(f'white panel found in {url}')
                skip_buoy_list.append(url)
                buoy_urls.remove(url)

                buoy_id = url.split('/')[-1].split('=')[-1]

                with open('scripts/failing_buoys.csv', 'a') as f:
                    f.write(f'{buoy_id}\n')
                    f.write(f'https://www.ndbc.noaa.gov/buoycam.php?station={buoy_id}\n')

                continue
            else:
                print(f'white panel not found in {url}')

        if any(c != "normal" for c in classifications) and classifications != [] and \
            all(c != "white" for c in classifications):

            c = max(set(classifications), key=classifications.count)
            x = "object"
            y = "white"
            all_white = all(y == x for x in classifications)
            any_c = any(c == x for x in classifications)

            if c == "storm" and "sunset" in classifications:
                c = "sunset"
            elif c == "normal" and "sunset" in classifications:
                c = "sunset"
            elif c == "normal" and "night" in classifications:
                c = "night"

            elif c == "object":
                c = "object"

            elif all_white:
                print(f'white panel found in {url}')
                skip_buoy_list.append(url)
                buoy_urls.remove(url)

                buoy_id = url.split('/')[-1].split('=')[-1]

                with open('scripts/failing_buoys.csv', 'a') as f:
                    f.write(f'{buoy_id}\n')
                    f.write(f'https://www.ndbc.noaa.gov/buoycam.php?station={buoy_id}\n')

                continue

            i = classifications.index(c)

            panel = panels[i]

            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            if save_confidence_plots:
                fig, ax = plt.subplots()
                ax.imshow(panel)
                text = f'{c} {np.max(classifications.count(c)) / len(classifications):.2f}'
                ax.annotate(text, (0, 0), (0, -20), xycoords='axes fraction', textcoords='offset points', va='top')
                ax.axis('off')
                ax.set_aspect('equal')
                fig.tight_layout()
                fig.canvas.draw()

                panel = np.array(fig.canvas.renderer._renderer)

                plt.close(fig)

                try:
                    panel = Image.fromarray(panel).resize((panel.width * upscalefactor, panel.height * upscalefactor), Image.BICUBIC)

                except AttributeError:

                    panel = Image.fromarray(panel).resize((panel.shape[1] * upscalefactor, panel.shape[0] * upscalefactor), Image.BICUBIC)

                    panel = panel.convert('RGB')

                panel.save(f"{url.split('/')[-2]}_{timestamp}_6x.jpg")

            panel_set = panels

            folder_name = str(c)
            os.makedirs(folder_name, exist_ok=True)

            if not only_save_originals:
                for ii, pan in enumerate(panel_set):

                    pan.save(os.path.join(folder_name, f"{ii}_[{c}]_{timestamp}.jpg"))

            image = image.resize((image.width * 4, image.height * 4), Image.BICUBIC)

            filename = f"{url.split('/')[-2]}_{timestamp}.jpg"
            filename = filename.replace(" ", "_")

            ext_filename = os.path.join(folder_name, filename).replace('/',"_")
            if not os.path.exists(ext_filename):
                image.save(os.path.join(folder_name, ext_filename))

def run_analysis_loop():
    """
    The run_analysis_loop function is the main function that runs the analysis.
    It first reads in a list of buoys from a csv file, and then loops through each buoy,
    downloading an image from it's url and running it through our model to get predictions.
    The results are saved as a csv file for later use.

    :return: A list of buoy urls
    :doc-author: Trelent
    """

    first_buoys = pd.read_csv("scripts/manual_buoys.csv")["station_id"].tolist()
    buoy_urls = pd.read_csv("scripts/working_buoys.csv")["station_id"].tolist()
    buoy_urls = list(dict.fromkeys(first_buoys + buoy_urls))

    with open('skip_buoy_list.txt', 'r') as f:
        skip_buoy_list = f.read().splitlines()

    skip_buoy_list = list(dict.fromkeys(skip_buoy_list))

    buoy_urls = [x for x in buoy_urls if x not in skip_buoy_list]

    while True:
        analyze_buoys(model, white_model, buoy_urls, save_confidence_plots=True, only_save_originals=False, white_mode=False)

        print('Waiting 10 minutes')
        time.sleep(600)

        if len(buoy_urls) == 0:
            print('No buoys left to process')
            break

def stitch_images(input_dir, output_path):
    """
    Stitches all images in a specified folder into one canvas that forms a square grid.

    :param input_dir: str, directory where images are stored
    :param output_path: str, path to save the stitched image
    """

    images = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]

    image_count = len(images)
    grid_size = int(math.sqrt(image_count))

    images = images[:grid_size**2]

    img_list = [Image.open(os.path.join(input_dir, img)).resize((100, 100)) for img in images]

    canvas = Image.new('RGB', (grid_size*100, grid_size*100))

    for i in range(grid_size):
        for j in range(grid_size):
            canvas.paste(img_list[i*grid_size + j], (j*100, i*100))

    canvas.save(output_path)

def balance_classes_for_model_training(input_dir, output_dir):
    """
    Balances classes for model training.

    :param input_dir: str, directory where original images are stored
    :param output_dir: str, directory where balanced images will be stored
    """

    classes = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]

    image_counts = [len(os.listdir(os.path.join(input_dir, c))) for c in classes]
    min_count = min(image_counts)

    for c in classes:
        class_dir = os.path.join(input_dir, c)
        output_class_dir = os.path.join(output_dir, c)

        os.makedirs(output_class_dir, exist_ok=True)

        images = [f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))]

        selected_images = random.sample(images, min_count)

        for image in selected_images:
            shutil.copy(os.path.join(class_dir, image), os.path.join(output_class_dir, image))

def main():
    """
    The main function is the entry point for this script.
    It will run a loop that downloads images from NOAA buoy cameras,
    classifies them using the model specified in load_model(), and then saves them to disk.

    :return: The output of the analysis loop
    :doc-author: Trelent
    """

    first_buoys = pd.read_csv("scripts/manual_buoys.csv")["station_id"].tolist()

    skip_buoy_list = []

    first_buoys = pd.read_csv("scripts/manual_buoys.csv")["station_id"].tolist()
    buoy_urls = pd.read_csv("scripts/working_buoys.csv")["station_id"].tolist()
    buoy_urls = first_buoys + buoy_urls

    buoy_urls = list(dict.fromkeys(buoy_urls))
    if len(refine_list(buoy_urls, skip_buoy_list)) > 0:
        refined_buoys = refine_list(buoy_urls, skip_buoy_list)
    else:
        refined_buoys = first_buoys
    buoy_urls = refined_buoys
    buoy_urls = [f'https://www.ndbc.noaa.gov/buoycam.php?station={buoy}' for buoy in buoy_urls]

    buoy_urls = list(dict.fromkeys(buoy_urls))

    try:
        if not os.path.exists('balanced_training_data'):
            os.makedirs('balanced_training_data')
        balance_classes_for_model_training('panels', 'balanced_training_data')
    except:
        print('Could not create balanced training data')

    try:

        stitch_images('BuoyEyeShots', 'BuoyEyeCanvas.png')
    except:
        print('Could not stitch images')

    model = load_model("models/gen3_keras/keras_model.h5")
    white_model = load_model("models/blank_or_not_model/keras_model.h5")

    run_analysis_loop()

if __name__ == "__main__":
    main()
