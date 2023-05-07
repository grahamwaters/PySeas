import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
# import ImageDraw
from PIL import ImageDraw
# pip install segmentation_models_pytorch

def split_into_panels(image, num_panels=6):
    width, height = image.size
    panel_width = width // num_panels
    panels = [image.crop((i * panel_width, 0, (i + 1) * panel_width, height)) for i in range(num_panels)]
    return panels


def crop_bottoms(panels, num_pixels=30):
    cropped_panels = [panel.crop((0, 0, panel.width, panel.height - num_pixels)) for panel in panels]
    return cropped_panels


def detect_horizon_tilt(panel):
    horizon_tilt = detect_horizon_line(panel)
    return horizon_tilt


def avg_color_above_horizon(panel, horizon_tilt):
    panel_array = np.array(panel.rotate(horizon_tilt))
    height = panel_array.shape[0] // 2
    sky_array = panel_array[:height]
    avg_sky_color = np.mean(sky_array, axis=(0, 1))
    return avg_sky_color


def avg_color_below_horizon(panel, horizon_tilt):
    panel_array = np.array(panel.rotate(horizon_tilt))
    height = panel_array.shape[0] // 2
    ocean_array = panel_array[height:]
    avg_ocean_color = np.mean(ocean_array, axis=(0, 1))
    return avg_ocean_color



import numpy as np
import cv2
from PIL import Image

def detect_horizon_line(img, resize_dim=None,
                        blur_kernel_size=5,
                        canny_low_ratio=0.33,
                        canny_high_ratio=1.33,
                        angle_tolerance=10):
    """
    The detect_horizon_line function takes an image and returns the angle of the horizon line.

    :param img: Pass in the image to be processed
    :param resize_dim: Increase the resolution of the image, type(tuple) (width, height)
    :param blur_kernel_size: Define the size of the kernel used in gaussian blur
    :param canny_low_ratio: Set the lower threshold for canny edge detection
    :param canny_high_ratio: Set the upper threshold for the canny edge detection algorithm
    :param angle_tolerance: Filter out lines that are not close to the horizontal orientation
    :return: The angle of the horizon line in degrees
    :doc-author: Trelent
    """

    # 1. Increase resolution (if resize_dim provided)
    if resize_dim:
        img = img.resize(resize_dim, Image.ANTIALIAS)

    img = np.array(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Blur and edge detection
    blurred = cv2.GaussianBlur(gray, (blur_kernel_size, blur_kernel_size), 0)

    # 3. Dynamic thresholding in Canny edge detection
    median_val = np.median(blurred)
    lower = int(max(0, canny_low_ratio * median_val))
    upper = int(min(255, canny_high_ratio * median_val))
    edges = cv2.Canny(blurred, lower, upper, apertureSize=3)

    # 4. HoughLinesP function parameters
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 150, minLineLength=100, maxLineGap=20)

    if lines is None:
        return 0

    try:
        horizontal_lines = []

        # 5. Filtering lines based on the angle
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi

            if abs(angle) <= angle_tolerance:
                horizontal_lines.append(line)

        if not horizontal_lines:
            return 0

        # Select the line closest to the horizontal orientation
        closest_horizontal_line = min(horizontal_lines, key=lambda line: abs(np.arctan2(line[0][3] - line[0][1], line[0][2] - line[0][0])))
        x1, y1, x2, y2 = closest_horizontal_line[0]
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        return angle

    except Exception as e:
        print(f"Error: {e}")
        return 0




def make_horizon_straight(img):
    # using the horizon tilt, rotate the image to make the horizon straight and return the rotated image
    horizon_tilt = detect_horizon_line(img,
                                       resize_dim=(3000, 1500),
                                        blur_kernel_size=5,
                                        canny_low_ratio=0.33,
                                        canny_high_ratio=1.33,
                                        angle_tolerance=10)
    print(f'Rotating panel {horizon_tilt} degrees')
    return img.rotate(horizon_tilt) #

def stitch_panels(panels):
    # Get the total width and height of the stitched panorama
    total_width = sum([panel.width for panel in panels])
    height = panels[0].height

    # Create a new image with the total width and the same height as the panels
    panorama = Image.new("RGB", (total_width, height))

    # Paste each panel into the new image
    current_x = 0
    for panel in panels:
        panorama.paste(panel, (current_x, 0))
        current_x += panel.width

    return panorama

def add_horizon_line(img, horizon_tilt):
    # add a red line where the horizon is detected, at the angle of the horizon tilt
    img = img.copy()
    draw = ImageDraw.Draw(img)
    width, height = img.size
    horizon_line = height / 2
    horizon_line = horizon_line + (horizon_tilt / 180) * height / 2
    draw.line((0, horizon_line, width, horizon_line), fill=(255, 0, 0), width=5)
    return img

def main():
    # Load and process the image
    panoramic_image = Image.open("W44A_2023_05_07_1710.jpg")
    panels = split_into_panels(panoramic_image)
    cropped_panels = crop_bottoms(panels)
    # use the make_horizon_straight function to make the horizon straight
    cropped_panels = [make_horizon_straight(panel) for panel in cropped_panels]
    horizon_tilts = [detect_horizon_tilt(panel) for panel in cropped_panels]
    avg_sky_colors = [avg_color_above_horizon(panel, tilt) for panel, tilt in zip(cropped_panels, horizon_tilts)]
    avg_ocean_colors = [avg_color_below_horizon(panel, tilt) for panel, tilt in
                        zip(cropped_panels, horizon_tilts)]

    # Add the horizon line to the panels
    cropped_panels = [add_horizon_line(panel, tilt) for panel, tilt in zip(cropped_panels, horizon_tilts)]

    # Plot the results
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    for i, panel in enumerate(cropped_panels):
        ax[i // 3, i % 3].imshow(panel.rotate(horizon_tilts[i]))
        ax[i // 3, i % 3].set_title(f"Panel {i + 1}")
        ax[i // 3, i % 3].axis("off")
    plt.show()
    #x = input("Press enter to continue")

    # use stitch_panels to stitch the panels together
    new_panorama = stitch_panels(cropped_panels)
    new_panorama.show()

    # save the new panorama
    new_panorama.save("new_panorama.jpg")

    print(f'Panorama saved as "new_panorama.jpg"')



if __name__ == "__main__":
    main()