import os
import cv2
import requests
from PIL import Image
from io import BytesIO
import numpy as np
from datetime import datetime
import cv2
import numpy as np

class BuoyImage:
    def __init__(self, image_url, num_panels=6, target_height=500):
        self.image_url = image_url
        self.image = self.download_image(image_url)
        self.panels = self.split_image_into_panels()
        self.unusual_panels = self.check_unusual_panels()
        self.image_url = image_url
        self.num_panels = num_panels
        self.target_height = target_height
        self.image = self.download_image(image_url)
        self.resized_image = self.resize_image_to_standard_height()
        self.panels = self.split_image_into_panels()

    def download_image(self, image_url):
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content))
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

    def mse_between_arrays(self, arr1, arr2):
        return np.mean((arr1 - arr2) ** 2)

    def check_unusual_panels(self):
        for i in range(len(self.panels) - 1):
            panel1 = np.asarray(self.panels[i])
            panel2 = np.asarray(self.panels[i + 1])
            mse = self.mse_between_arrays(panel1, panel2)

            if mse > self.mse_threshold:
                return True

        return False

class PanoramicImage:
    def __init__(self, images):
        self.images = images
        self.aligned_images = [self.align_horizon_line(img) for img in images]
        self.stitched_image = self.stitch_aligned_images()

    def detect_horizon_line(self, img):
        # Convert the image to grayscale and apply a Canny edge detector
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        # Use HoughLinesP to detect lines in the image
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)

        # Find the longest line and calculate its tilt in degrees
        longest_line = max(lines, key=lambda line: np.linalg.norm(line[0][2:] - line[0][:2]))
        x1, y1, x2, y2 = longest_line[0]
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        return angle

    def align_horizon_line(self, img):
        # Calculate the tilt of the horizon line
        tilt_angle = self.detect_horizon_line(img)

        # Rotate the image to align the horizon line
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
    buoy_urls = ["buoy_url_1", "buoy_url_2", "buoy_url_3"]
    base_output_path = "path/to/output/directory"

    for buoy_url in buoy_urls:
        # Download the image and create a BuoyImage instance
        buoy_image = BuoyImage(buoy_url)

        if any(buoy_image.unusual_panels):
            # Save the panels in a folder named after the buoy > date > time
            current_datetime = datetime.now()
            date_str = current_datetime.strftime("%Y-%m-%d")
            time_str = current_datetime.strftime("%H-%M-%S")
            buoy_name = buoy_url.split("/")[-2]  # Extract buoy name from URL

            output_path = os.path.join(base_output_path, buoy_name, date_str, time_str)
            os.makedirs(output_path, exist_ok=True)

            for i, panel in enumerate(buoy_image.panels):
                panel_output_path = os.path.join(output_path, f"panel_{i+1}.jpg")
                panel.save(panel_output_path)

            # Stitch the panels back together, aligning the horizon lines if possible
            panoramic_image = PanoramicImage(buoy_image.panels)
            panoramic_image.align_horizon_lines()

            panorama_output_path = os.path.join(output_path, "panorama.jpg")
            panoramic_image.panorama.save(panorama_output_path)

if __name__ == "__main__":
    main()
