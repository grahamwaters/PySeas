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
# Import necessary libraries
import requests
import cv2
import numpy as np
import os
from helper_functions import make_panorama, artist_eval, display_last_image, crop_the_bottom_off, get_panel_segments, stitched_panoramas


import numpy as np
import cv2
import os
import imutils
from PIL import Image

class enVitArtist:
    def __init__(self):
        self.name = "enVitArtist"
        self.buoys = []
        self.weather_conditions = []
        self.image_data = []
        self.stitched_image_data = []
        self.horizon_line = []
        self.time_lapse_data = []

    def artist_eval(self, image_path):
        img = Image.open(image_path)
        width, height = img.size
        panels = [img.getpixel((int(width * i / 12), int(height / 2))) for i in [1, 3, 6, 9, 10, 11]]
        mses = [np.mean((np.array(panels[i]) - np.array(panels[i + 1])) ** 2) for i in range(len(panels) - 1)]
        mse = np.mean(mses)

        return mse < 100

    def make_panorama(self, images):
        """
        make_panorama stitches together a list of images into a single panoramic image. It uses the OpenCV library to do this.

        :param images: this is a list of images that we want to stitch together into a single panoramic image
        :type images: list of strings (file paths)
        :return: the stitched image if the stitching was successful, otherwise None
        :rtype: numpy array
        """
        scale_percent = 70
        if '.DS_Store' in images:
            images.remove('.DS_Store')
        images_opened = [cv2.resize(cv2.imread(image), None, fx=scale_percent / 100, fy=scale_percent / 100, interpolation=cv2.INTER_AREA) for image in images]
        stitcher = cv2.createStitcher() if imutils.is_cv3() else cv2.Stitcher_create()
        stitcher.setPanoConfidenceThresh(0.1)
        stitcher.setRegistrationResol(0.6)
        stitcher.setSeamEstimationResol(0.1)
        stitcher.setCompositingResol(0.6)
        stitcher.setFeaturesFinder(cv2.ORB_create())
        stitcher.setFeaturesMatcher(cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING))
        stitcher.setBundleAdjuster(cv2.detail_BundleAdjusterRay())
        stitcher.setExposureCompensator(cv2.detail.ExposureCompensator_createDefault(cv2.detail.ExposureCompensator_GAIN_BLOCKS))
        stitcher.setBlender(cv2.detail.Blender_createDefault(cv2.detail.Blender_NO))
        stitcher.setSeamFinder(cv2.detail.SeamFinder_createDefault(cv2.detail.SeamFinder_VORONOI_SEAM))
        stitcher.setWarper(cv2.PyRotationWarper(cv2.ROTATION_WARP_PERSPECTIVE))
        stitcher.setInterpolationFlags(cv2.INTER_LINEAR_EXACT)
        status, stitched = stitcher.stitch(images_opened)

        if status == 0:
            stitched = cv2.resize(stitched, None, fx=scale_percent / 100, fy=scale_percent / 100, interpolation=cv2.INTER_AREA)
            return stitched
        else:
            return None

    def get_average_color(self, image):
        """
        get_average_color determines the average color of the image in the center of the image by pixel.

        :param image: this is the image that we want to get the average color of, it can be a numpy array or an image object
        :type image: numpy array or image object
        :return: the average color of the image in the center of the image by pixel
        :rtype: tuple
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        img_width, img_height = image.size
        average_color = image.getpixel((img_width // 2, img_height // 2))

        return average_color




# Define the BuoyImage class
class BuoyImage:
    def __init__(self, location, weather_conditions, image_data):
        self.location = location
        self.weather_conditions = weather_conditions
        self.image_data = image_data

    def get_images(self):
        # Retrieve the images from the NOAA API
        pass

    def stitch_images(self):
        # Stitch the images together
        pass

    def blend_images(self):
        # Blend the images over time
        pass

# Define the PanoramicImage class
class PanoramicImage:
    def __init__(self, stitched_image_data, horizon_line, time_lapse_data):
        self.stitched_image_data = stitched_image_data
        self.horizon_line = horizon_line
        self.time_lapse_data = time_lapse_data

    def blend_images(self):
        # Blend the images over time
        pass

    def detect_horizon(self):
        # Detect the horizon line
        pass

    def create_time_lapse(self):
        # Create a time-lapse animation
        pass

# Define the main function
def main():
    # Instantiate the BuoyImage class
    buoy_image = BuoyImage(location, weather_conditions, image_data)

    # Retrieve images from NOAA API
    buoy_image.get_images()

    # Stitch images together
    buoy_image.stitch_images()

    # Instantiate the PanoramicImage class
    panoramic_image = PanoramicImage(stitched_image_data, horizon_line, time_lapse_data)

    # Detect the horizon line
    panoramic_image.detect_horizon()

    # Create a time-lapse animation
    panoramic_image.create_time_lapse()

    # Save the final output
    pass

# Call the main function
if __name__ == "__main__":
    main()
