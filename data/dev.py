import glob
import os
import cv2
import numpy as np
from tqdm import tqdm
import time

# Function to determine if an image is mostly black
def is_mostly_black(image, threshold=10):
    return np.mean(image) < threshold

# Function to stitch images vertically
def stitch_vertical(images):
    # Determine the width and height for the collage
    width = max(image.shape[1] for image in images)
    height = sum(image.shape[0] for image in images)

    # Create a blank canvas for the collage
    collage = np.zeros((height, width, 3), dtype=np.uint8)

    y_offset = 0
    for image in tqdm(images):
        collage[y_offset:y_offset+image.shape[0], :image.shape[1]] = image
        y_offset += image.shape[0]

    return collage


import cv2
import numpy as np

def split_into_panels(image, number_of_panels=6):
    # Split the image into six equal panels vertically (side by side)
    height, width = image.shape[:2]
    panel_width = width // number_of_panels
    panels = [image[:, i*panel_width:(i+1)*panel_width] for i in range(number_of_panels)]
    return panels


def remove_bottom_strip(panel, strip_height=30):
    # Remove the bottom strip from the panel
    return panel[:-strip_height]

def enhance_image(panel, clip_limit=2.0, tile_grid_size=(8, 8)):
    # Convert to YUV color space
    panel_yuv = cv2.cvtColor(panel, cv2.COLOR_BGR2YUV)
    # Apply CLAHE to the Y channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    panel_yuv[:, :, 0] = clahe.apply(panel_yuv[:, :, 0])
    # Convert back to BGR color space
    enhanced_panel = cv2.cvtColor(panel_yuv, cv2.COLOR_YUV2BGR)
    return enhanced_panel


def preprocess_and_stitch_panels(image, number_of_panels=6, strip_height=20):
    # Split the image into panels
    panels = split_into_panels(image, number_of_panels)

    # Process each panel
    processed_panels = []
    for panel in tqdm(panels):
        # Remove the bottom strip
        panel_without_strip = remove_bottom_strip(panel, strip_height)

        # Enhance the panel
        # enhanced_panel = enhance_image(panel_without_strip)
        try:
            processed_panels.append(enhanced_panel)
        except Exception as e:
            print(e)
            processed_panels.append(panel_without_strip)

    # Stitch the panels back together vertically
    return np.concatenate(processed_panels, axis=0)

# Example usage:
# image = cv2.imread('path_to_your_image.jpg')
# processed_image = preprocess_and_stitch_panels(image)
# Now this processed_image can be passed to the stitch_vertical function to create a collage.

# Filtering images based on orange value and blackness
files = glob.glob('images/buoys/*/*')
latest_images = []  # A new list to hold processed images
latest_image_files = []  # A new list to hold filenames of images that pass the filter
for file in tqdm(files):
    image = cv2.imread(file)
    orange_value = np.mean(image[:,:,2])

    # Check conditions for orange value and blackness
    if not (10 <= orange_value <= 150) or is_mostly_black(image):
        continue
    # Check if the image is too similar to the previous one by comparing filenames
    elif len(latest_image_files) > 0 and np.array_equal(image, cv2.imread(latest_image_files[-1])):
        continue
    else:
        processed_image = preprocess_and_stitch_panels(image)
        latest_images.append(processed_image)  # Append the processed image (not the filename)
        latest_image_files.append(file)  # Append the filename to the new list


# Now `latest_images` contains the processed images ready for vertical stitching
collage = stitch_vertical(latest_images)


# Save the collage to disk as a single image with the current timestamp
timestamp = str(int(time.time()))
collage_directory = 'images/save_images'
os.makedirs(collage_directory, exist_ok=True)  # Ensure the directory exists
filename = os.path.join(collage_directory, f'collage_{timestamp}.jpg')
cv2.imwrite(filename, collage)
print(f"Collage saved to {filename}")

# Depending on your environment, the cv2.imshow might not work. In such cases, you can remove these lines.
# Display the collage
cv2.imshow('Vertical Collage', collage)
cv2.waitKey(0)
cv2.destroyAllWindows()
