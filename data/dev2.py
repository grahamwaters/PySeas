import glob
import os
import cv2
import numpy as np
from tqdm import tqdm
import time

# Function to determine if an image is mostly black
def is_mostly_black(image, threshold=10):
    return np.mean(image) < threshold

# Function to stitch images horizontally
def stitch_panels_horizontally(panels):
    # Ensure all panels are the same height before stitching
    max_height = max(panel.shape[0] for panel in panels)
    panels_resized = [cv2.resize(panel, (panel.shape[1], max_height), interpolation=cv2.INTER_LINEAR) for panel in panels]
    return np.concatenate(panels_resized, axis=1)

# Function to stitch images vertically
def stitch_vertical(rows):
    # Ensure all rows are the same width before stitching
    max_width = max(row.shape[1] for row in rows)
    # Resize rows to the max width or pad with black pixels
    rows_resized = []
    for row in rows:
        if row.shape[1] < max_width:
            padding = np.zeros((row.shape[0], max_width - row.shape[1], 3), dtype=np.uint8)
            row_resized = np.concatenate((row, padding), axis=1)
        else:
            row_resized = row
        rows_resized.append(row_resized)

    # Stitch the rows together
    return np.concatenate(rows_resized, axis=0)
# Split into vertical panels
def split_into_panels(image, number_of_panels=6):
    # Split the image into six equal vertical panels (side by side)
    width = image.shape[1]
    panel_width = width // number_of_panels
    panels = [image[:, i*panel_width:(i+1)*panel_width] for i in range(number_of_panels)]
    # Ensure last panel takes any remaining pixels to account for rounding
    panels[-1] = image[:, (number_of_panels-1)*panel_width:]
    return panels

# Remove the bottom strip from each panel
def remove_bottom_strip(panel, strip_height=20):
    # Assume the strip to be removed is at the bottom of the image
    return panel[:-strip_height, :]

# Enhance the image
def enhance_image(panel, clip_limit=2.0, tile_grid_size=(8, 8)):
    # Apply any enhancements to the panel, like histogram equalization, etc.
    # Convert to YUV color space
    panel_yuv = cv2.cvtColor(panel, cv2.COLOR_BGR2YUV)
    # Apply CLAHE to the Y channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    panel_yuv[:, :, 0] = clahe.apply(panel_yuv[:, :, 0])
    # Convert back to BGR color space
    enhanced_panel = cv2.cvtColor(panel_yuv, cv2.COLOR_YUV2BGR)
    return enhanced_panel

# Process and stitch panels
def preprocess_and_stitch_panels(image, number_of_panels=6, strip_height=35):
    panels = split_into_panels(image, number_of_panels)
    processed_panels = [enhance_image(remove_bottom_strip(panel, strip_height)) for panel in panels]
    return stitch_panels_horizontally(processed_panels)

# Main processing logic
files = glob.glob('images/buoys/*/*')
rows_to_stitch = []
latest_image_files = []  # Maintain a list of filenames that have been processed

for file in tqdm(files):
    image = cv2.imread(file)
    orange_value = np.mean(image[:,:,2])

    if not (10 <= orange_value <= 150) or is_mostly_black(image):
        continue
    # Compare with the last image file processed, not the processed row
    elif latest_image_files and np.array_equal(image, cv2.imread(latest_image_files[-1])):
        continue

    row = preprocess_and_stitch_panels(image)
    rows_to_stitch.append(row)  # Append the row for stitching
    latest_image_files.append(file)  # Append the file to the list for comparison


# Create a collage from the rows
collage = stitch_vertical(rows_to_stitch)

# Save the collage with timestamp
timestamp = str(int(time.time()))
collage_directory = 'images/save_images'
os.makedirs(collage_directory, exist_ok=True)
filename = os.path.join(collage_directory, f'collage_{timestamp}.jpg')
cv2.imwrite(filename, collage)
print(f"Collage saved to {filename}")

# Display the collage
cv2.imshow('Vertical Collage', collage)
cv2.waitKey(0)
cv2.destroyAllWindows()
