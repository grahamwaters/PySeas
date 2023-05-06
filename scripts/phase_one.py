"""
PySeas - Phase One: Sunrise over the Sea
In Phase One, we will focus on retrieving images of sunsets over the ocean using the NOAA API, stitching them together to create a single panoramic image, and finally generating a time-lapse of the sunset.

Outline of the Python Script
Import necessary libraries
Define the BuoyImage class
Define the PanoramicImage class
Define the main function
Call the main function

Logic Outline:
- For each buoy, retrieve the latest image from the NOAA website using the url pattern provided by the API
- Split the image into the panels there should be 6 panels horizontally divided. The bottom 30 px should be discarded.

"""
import requests
import cv2
import numpy as np
import os
import math
import imutils
from tqdm import tqdm
from PIL import Image
from io import BytesIO
from datetime import datetime
from ratelimit import limits, sleep_and_retry
from matplotlib import pyplot as plt
import time
import random
import pandas as pd
# Configuration
collecting_all = True
buoy_list = pd.read_csv("scripts/working_buoys.csv")
# Utility functions
def mse_between_arrays(arr1, arr2):
    return np.mean((arr1 - arr2) ** 2)


# BuoyImage class
class BuoyImage:
    def __init__(self, image_url, num_panels=6, target_height=500, mse_threshold=2000):
        self.image_url = image_url
        self.num_panels = num_panels
        self.target_height = target_height
        self.mse_threshold = mse_threshold
        self.image = self.download_image(image_url)
        self.resized_image = self.resize_image_to_standard_height()
        self.panels = self.split_image_into_panels()

    @sleep_and_retry
    def download_image(self, image_url):
        global buoy_list
        # time.sleep(random.randint(1, 3))
        response = requests.get(image_url)
        if response.status_code != 200:
            print(f"API response: {response.status_code}")
        img = Image.open(BytesIO(response.content))
        img_array = np.asarray(img)
        if np.sum(img_array > 200) / img_array.size < 0.9: # if more than 90% of the image is dark
            print(f"Image too dark: {image_url}")
            # drop image if too dark (from the csv file)
            with open("scripts/failing_buoys.csv", "r") as f:
                failing_buoys = f.read().splitlines()
            if image_url not in failing_buoys:
                with open("scripts/failing_buoys.csv", "a") as f:
                    f.write(image_url + "\n")
            # drop image from the csv file #TODO
            # extract buoy from image_url
            buoy = image_url.split("/")[-2]
            # drop buoy from the csv file
            buoy_list = [b for b in buoy_list if b != buoy]
            # save the updated csv file

            buoy_list.to_csv("scripts/working_buoys.csv", index=False)
            return None
        return img

    def resize_image_to_standard_height(self):
        width, height = self.image.size
        new_height = self.target_height
        new_width = int((new_height / height) * width)
        return self.image.resize((new_width, new_height), Image.ANTIALIAS)

    def split_image_into_panels(self):
        width, height = self.resized_image.size
        panel_width = width // self.num_panels

        panels = []
        for i in range(self.num_panels):
            left = i * panel_width
            right = left + panel_width
            panel = self.resized_image.crop((left, 0, right, height))
            panels.append(panel)

        return panels

    def check_unusual_panels(self, panel_set=[]):
        unusual_panels = []
        rich_color_panels = []
        panel_set = panel_set if panel_set else self.panels
        panel_mses = []
        for i, panel in enumerate(panel_set):
            # Convert panel to an numpy array for MSE calculation
            panel_array = np.array(panel)
            # Calculate MSE between panel and itself
            panel_mse = mse_between_arrays(panel_array, panel_array)
            panel_mses.append(panel_mse)
        # Calculate median MSE for the panel set
        median_mse = np.median(panel_mses)
        # Calculate a threshold for rich color based on the median MSE
        rich_color_threshold = median_mse * 0.8
        for i, panel in enumerate(panel_set):
            panel_array = np.array(panel)
            panel_mse = panel_mses[i]
            if panel_mse > self.mse_threshold:
                unusual_panels.append(i)
            if panel_mse > rich_color_threshold:
                rich_color_panels.append(panel)
        return unusual_panels, rich_color_panels



class PanoramicImage:
    def __init__(self, images):
        self.images = images
        self.aligned_images = [self.align_horizon_line(img) for img in images]
        self.stitched_image = self.stitch_aligned_images()

    @staticmethod
    def detect_horizon_line(img):
        img = np.array(img)
        # show a rough estimate of the horizon line in matplotlib
        # if the img is not over 90% white pixels then show the horizon line
        if np.sum(img > 200) / img.size < 0.9:
            # then show the horizon line
            plt.imshow(img)
            plt.axhline(y=30, color='r', linestyle='-')
            # save the plot as `latest_horizon_line.png`
            plt.savefig('latest_horizon_line.png')
        else:
            # save a blank plot
            pass
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
        try:
            longest_line = max(lines, key=lambda line: np.linalg.norm(line[0][2:] - line[0][:2]))
            x1, y1, x2, y2 = longest_line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            return angle
        except TypeError:
            # station may not be online or have a buoycam
            return 0
        except Exception as e:
            print(f"Error: {e}")
            return 0
    def align_horizon_line(self, img):
        img = np.array(img)
        tilt_angle = PanoramicImage.detect_horizon_line(img)
        height, width = img.shape[:2]
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, tilt_angle, 1)
        aligned_img = cv2.warpAffine(img, rotation_matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return aligned_img

    def stitch_aligned_images(self):
        stitcher = cv2.createStitcher() if imutils.is_cv3() else cv2.Stitcher_create()
        (status, stitched_image) = stitcher.stitch(self.aligned_images)

        if status == 0:
            return stitched_image
        else:
            return None


def main():

    buoy_urls = [f'https://www.ndbc.noaa.gov/buoycam.php?station={station}' for station in buoy_list]
    base_output_path = "images/output_test"

    for buoy_url in tqdm(buoy_urls, desc="Processing buoys"):
        print(f'Processing {buoy_url}', end='', flush=True)
        buoy_image = BuoyImage(buoy_url)
        # test if any panels are unusual in the image
        panel_set = buoy_image.panels
        unusual_panels = buoy_image.check_unusual_panels(panel_set)

        if unusual_panels:
            print(f" - Unusual panels: {unusual_panels}")
            current_datetime = datetime.now()
            date_str = current_datetime.strftime("%Y-%m-%d")
            time_str = current_datetime.strftime("%H-%M-%S")
            buoy_name = buoy_url.split("/")[-2]

            output_path = os.path.join(base_output_path, buoy_name, date_str, time_str)
            os.makedirs(output_path, exist_ok=True)

            for i, panel in enumerate(buoy_image.panels):
                panel_output_path = os.path.join(output_path, f"panel_{i+1}.jpg")
                panel.save(panel_output_path)

            panoramic_image = PanoramicImage(buoy_image.panels)

            panorama_output_path = os.path.join(output_path, "panorama.jpg")
            try:
                cv2.imwrite(panorama_output_path, panoramic_image.stitched_image)
                # update this image by saving it to the image `latest.png` in the main directory for easy viewing
                latest_output_path = os.path.join(base_output_path, buoy_name, "latest.jpg")
                cv2.imwrite(latest_output_path, panoramic_image.stitched_image)
            except AttributeError:
                print("Panorama could not be stitched")
            except Exception as e:
                print(f"Error: {e}, with {buoy_url}")
        else:
            print(" - No unusual panels")

if __name__ == "__main__":
    main()
