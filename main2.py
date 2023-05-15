import os
import time
import requests
from PIL import Image
from io import BytesIO
from datetime import datetime
from image_classifier import ImageClassifier
import pandas as pd
from tensorflow.keras.models import load_model
from tqdm import tqdm
import numpy as np
from PIL import ImageDraw, ImageFont, Image
MODEL_PATH = 'models/converted_keras/keras_model.h5'
BLANK_OR_NOT_MODEL_PATH = 'models/blank_or_not_model/keras_model.h5'
white_mode = False
only_save_originals = True
save_confidence_plots = False
from colorama import Fore, Back, Style
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

def refine_list(buoy_urls, skip_buoy_list):
    # using the prefixes of the files in the /raw folder, build a list of non-malfunctioing buoys
    # get the list of files in the /raw folder
    files = os.listdir('raw')
    # get the list of buoys from the files
    buoys = [file.split('_')[0] for file in files]
    # remove duplicates
    buoys = list(dict.fromkeys(buoys))
    return buoys

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
    avg_color = np.array(panel).mean(axis=(0, 1))
    if avg_color.mean() > 200:
        return 'white'
    elif avg_color.mean() < 10:
        return 'night'
    else:
        return ImageClassifier.classify_image(panel, model, white_model)

def analyze_buoys(model, white_model):
    classifier = ImageClassifier()

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

        avg_color = np.array(image).mean(axis=(0,1))
        if avg_color.mean() > 300:
            continue
        else:

            image.save(f'raw/{url.split("/")[-1].split("=")[-1]}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        panels = process_image(image)

        classifications = [classifier.classify_image(panel, model, white_model) for panel in panels]

        for panel in panels:
            white_mode = False #TODO
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
            # if there are 6 panels and the average color of each panel is less than 250, then classify the image
            if len(panels) == 6 and panel_avg_color.mean() < 250:
                classification = classifier.classify_image(panel, model)
                # save the panels in a {classification} folder for later review, if there is none then create one
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
                continue
        for i, panel in enumerate(panels):
            panel_annotated = panel.copy()
            draw = ImageDraw.Draw(panel_annotated)
            # annotate the image with the classification in red text
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
                    panel = Image.fromarray(panel).resize((panel.width * 6, panel.height * 6), Image.BICUBIC)
                except AttributeError:

                    panel = Image.fromarray(panel).resize((panel.shape[1] * 6, panel.shape[0] * 6), Image.BICUBIC)

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
    first_buoys = pd.read_csv("scripts/manual_buoys.csv")["station_id"].tolist()
    buoy_urls = pd.read_csv("scripts/working_buoys.csv")["station_id"].tolist()
    buoy_urls = list(dict.fromkeys(first_buoys + buoy_urls))

    while True:
        analyze_buoys(model, white_model)

        print('Waiting 10 minutes')
        time.sleep(600)

        skip_buoy_list = pd.read_csv("scripts/failing_buoys.csv")["station_id"].tolist()
        skip_buoy_list = list(dict.fromkeys(skip_buoy_list))
        buoy_urls = [buoy for buoy in buoy_urls if buoy not in skip_buoy_list]

        if len(buoy_urls) == 0:
            print('No buoys left to process')
            break


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
buoy_urls = refined_buoys
buoy_urls = [f'https://www.ndbc.noaa.gov/buoycam.php?station={buoy}' for buoy in buoy_urls]
# remove duplicates
buoy_urls = list(dict.fromkeys(buoy_urls))




run_analysis_loop()
