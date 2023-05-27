import os
import requests
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
from datetime import datetime
from matplotlib import pyplot as plt
import cv2
# imutils
import imutils

class BuoyImage:
    def __init__(self, image_url, num_panels=6, target_height=500, mse_threshold=2000):
        self.image_url = image_url
        self.num_panels = num_panels
        self.target_height = target_height
        self.mse_threshold = mse_threshold
        self.image = self.download_image(image_url)
        self.resized_image = self.resize_image_to_standard_height()
        self.panels = self.split_image_into_panels()
        self.unusual_panels = self.check_unusual_panels()

    def download_image(self, image_url):
        """
        download_image downloads an image from a url and returns a PIL Image object

        :param image_url: url of image to download
        :type image_url: str
        :return: PIL Image object
        :rtype: PIL.Image.Image
        """
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content))
        return img

    def resize_image_to_standard_height(self):
        """
        The resize_image_to_standard_height function takes an image and resizes it to a standard height.
        The function returns the resized image.

        :param self: Represent the instance of the class
        :return: A resized image with a height of self
        :doc-author: Trelent
        """

        width, height = self.image.size
        new_height = self.target_height
        new_width = int((new_height / height) * width)
        return self.image.resize((new_width, new_height), Image.ANTIALIAS)

    def split_image_into_panels(self):
        """
        The split_image_into_panels function takes an image and splits it into a number of panels.
        The function returns a list of PIL images, each representing one panel.

        :param self: Refer to the object itself
        :return: A list of panels
        :doc-author: Trelent
        """

        width, height = self.resized_image.size
        panel_width = width // self.num_panels

        panels = []
        for i in range(self.num_panels):
            left = i * panel_width
            right = left + panel_width
            panel = self.resized_image.crop((left, 0, right, height))
            panels.append(panel)

        return panels

    def mse_between_arrays(self, arr1, arr2):
        """
        The mse_between_arrays function takes two arrays as input and returns the mean squared error between them.
            The function is used to compare the difference in pixel values between two images.

        :param self: Allow the function to access variables that are a part of the class
        :param arr1: Represent the first array of data
        :param arr2: Compare the values in arr2 to the values in arr 1
        :return: The mean squared error between two arrays
        :doc-author: Trelent
        """

        return np.mean((arr1 - arr2) ** 2)

    def check_unusual_panels(self):
        """
        The check_unusual_panels function takes in a list of panels and compares each panel to the next one.
        If the mean squared error between two panels is greater than a threshold, then it returns True for that panel.
        Otherwise, it returns False.

        :param self: Allow an object to refer to itself inside of a method
        :return: A list of booleans
        :doc-author: Trelent
        """

        unusual_panels = []
        for i in range(len(self.panels) - 1):
            panel1 = np.asarray(self.panels[i])
            panel2 = np.asarray(self.panels[i + 1])
            mse = self.mse_between_arrays(panel1, panel2)
            unusual_panels.append(mse > self.mse_threshold)

        return unusual_panels


class PanoramicImage:
    def __init__(self, images):
        self.images = images
        self.aligned_images = [self.align_horizon_line(img) for img in images]
        self.stitched_image = self.stitch_aligned_images()

    def detect_horizon_line(self, img):
        """
        The detect_horizon_line function takes an image as input and returns the angle of the horizon line.
        The function uses OpenCV to detect edges in a grayscale version of the image, then uses HoughLinesP to find lines in those edges.
        It then finds the longest line and calculates its angle using arctan2.

        :param self: Represent the instance of the class
        :param img: Pass the image to the function
        :return: The angle of the horizon line in degrees
        :doc-author: Trelent
        """

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)

        longest_line = max(lines, key=lambda line: np.linalg.norm(line[0][2:] - line[0][:2]))
        x1, y1, x2, y2 = longest_line[0]
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        return angle

    def align_horizon_line(self, img):
        """
        The align_horizon_line function takes an image and returns a new image with the horizon line aligned.
            The function uses the detect_horizon_line function to determine how much to rotate the original image.
            It then rotates it by that amount, using OpenCV's warpAffine method.

        :param self: Represent the instance of the class
        :param img: Pass the image to be aligned
        :return: The aligned image
        :doc-author: Trelent
        """

        tilt_angle = self.detect_horizon_line(img)
        height, width = img.shape[:2]
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, tilt_angle, 1)
        aligned_img = cv2.warpAffine(img, rotation_matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return aligned_img

    def stitch_aligned_images(self):
        """
        The stitch_aligned_images function takes a list of aligned images and stitches them together.
            The function returns the stitched image if successful, otherwise it returns None.

        :param self: Represent the instance of the class
        :return: The stitched image
        :doc-author: Trelent
        """

        stitcher = cv2.createStitcher() if imutils.is_cv3() else cv2.Stitcher_create()
        (status, stitched_image) = stitcher.stitch(self.aligned_images)

        if status == 0:
            return stitched_image
        else:
            return None


def main():
    buoy_urls = ["buoy_url_1", "buoy_url_2", "buoy_url_3"]
    base_output_path = "path/to/output/directory"

    for buoy_url in buoy_urls:
        buoy_image = BuoyImage(buoy_url)

        if any(buoy_image.unusual_panels):
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
            cv2.imwrite(panorama_output_path, panoramic_image.stitched_image)


if __name__ == "__main__":
    main()
