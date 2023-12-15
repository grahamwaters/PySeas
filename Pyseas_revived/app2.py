import os
import requests
from bs4 import BeautifulSoup
from flask import Flask, render_template
from PIL import Image
from io import BytesIO
import cv2
import numpy as np

def stitch_images(image_paths):
    """
    Stitch images together based on the horizon line.

    Args:
        image_paths (list): List of paths to the input images.

    Returns:
        numpy.ndarray: Stitched panorama image.
    """
    # Load the first image
    first_image = cv2.imread(image_paths[0])
    stitched_image = first_image

    # Iterate over the remaining images
    for i in range(1, len(image_paths)):
        # Load the current image
        current_image = cv2.imread(image_paths[i])

        # Detect the keypoints and compute the descriptors for the images
        sift = cv2.SIFT_create()
        keypoints1, descriptors1 = sift.detectAndCompute(stitched_image, None)
        keypoints2, descriptors2 = sift.detectAndCompute(current_image, None)

        # Match the keypoints using a FLANN based matcher
        matcher = cv2.FlannBasedMatcher()
        matches = matcher.knnMatch(descriptors1, descriptors2, k=2)

        # Apply ratio test to filter good matches
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        # Sort the matches by distance
        good_matches = sorted(good_matches, key=lambda x: x.distance)

        # Extract the matched keypoints from both images
        matched_keypoints1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches])
        matched_keypoints2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches])

        # Estimate the homography matrix using RANSAC
        homography, _ = cv2.findHomography(matched_keypoints2, matched_keypoints1, cv2.RANSAC)

        # Warp the current image to align with the stitched image
        warped_image = cv2.warpPerspective(current_image, homography, (stitched_image.shape[1], stitched_image.shape[0]))

        # Blend the warped image with the stitched image
        stitched_image = cv2.addWeighted(stitched_image, 0.5, warped_image, 0.5, 0)

    return stitched_image

app = Flask(__name__)

def get_buoy_list():
    return ['41001', '41002', '41003', '41004']

def get_buoy_images(buoy_ids):
    image_paths = []
    base_url = "http://www.ndbc.noaa.gov/buoycam.php?station="
    for buoy_id in buoy_ids:
        url = base_url + buoy_id
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        img_tag = soup.find('img', {"id": "latest_img"})
        if img_tag:
            image_url = 'http://www.ndbc.noaa.gov' + img_tag['src'] if img_tag['src'].startswith('/') else img_tag['src']
            response = requests.get(image_url)
            if response.status_code == 200:
                image = Image.open(BytesIO(response.content))
                image_path = f"temp/{buoy_id}.jpg"  # Save the image to a temporary file
                image.save(image_path)
                image_paths.append(image_path)
    return image_paths

@app.route('/')
def index():
    all_images = []
    buoy_ids = get_buoy_list()
    image_urls = get_buoy_images(buoy_ids)
    for buoy_id, image_url in image_urls:
        # Download the image
        response = requests.get(image_url)
        if response.status_code == 200:
            image = Image.open(BytesIO(response.content))
            all_images.append(image)

    # Once all images are collected, pass them to the stitch_images function
    panorama_image_path = stitch_images(all_images)

    return render_template('index.html', panorama_image=panorama_image_path)

if __name__ == '__main__':
    app.run(debug=True)
