Graham Waters
Project Pisces I want to make a beautiful, evolving art piece out of the images generated every hour by the buoys in the ocean, which the National Oceanographic Atmospheric Association or NOAA maintains. They have an API that allows you to call that API and pull images from buoys worldwide. I want to be able to take those images every hour from the buoys and stitch them together using Open CV in a way that makes it I would like to be able to stitch them together to create a panoramic photo, but not by just concatenating the images because that doesn't always look good. Also, I would like to match the images' horizon lines because the buoy images are uneven due to the waves. It is usually rocking back and forth. Another goal of this project is to find the most beautiful sunset panoramic photos and display those on the main project page. So that involves a little computer vision and classifying images as sunsets. I have some code already and just need to refactor it.

To refactor your code to create panoramic photos using OpenCV, you can follow these steps:

Use the NOAA API to pull images from the buoys every hour.
Use OpenCV's stitching module to automatically stitch the images together. The stitching module can align the images and blend them seamlessly to create a panoramic photo.
Use OpenCV's computer vision algorithms to detect and classify the images as sunsets. This can be done using a pre-trained machine learning model that has been trained to recognize sunsets in images.
Once the sunset images have been identified, display them on the main project page for others to see.
Overall, the key steps in this project involve using the NOAA API to pull images from the buoys, using OpenCV to stitch the images together and classify them, and displaying the resulting panoramic photos and sunset images on the main project page.




Graham Waters
from cam_backend import *

print("Starting the download loop")
last_time_fetched = time.time()  # get the current time
first_run = True  # set a flag to indicate that this is the first run of the loop (for the first run, we will download rss feeds for all the buoys)
duplicate_removal_flag = (
    True  # set this flag to true if we want to remove duplicated images with difPy
)
# note: bugs are present in difPy, so this flag is set to false
exper_1 = False  # flag for dupe detect in panels
verbose_wait = False  # flag that makes waiting show a progress bar.

# import the necessary packages
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import glob

rotating = True  # flag to indicate if the tapestry is rotating
panel_mode = False  # flag to indicate if we want to use panels for color detection


def is_recent(file, minutes):
    # get the time the image was taken
    image_time = os.path.getmtime(file)
    # get the current time
    current_time = time.time()
    # get the difference between the two times
    time_difference = current_time - image_time
    # if the time difference is less than minutes, return true
    if time_difference < minutes * 60:
        return True
    else:
        return False


def crop_the_bottom_off(images):
    # for each of the images crop the bottom off (20 pixels)
    for image in images:
        try:
            # get the image size
            img_width, img_height = get_image_size(image)
            # crop the bottom off
            cropped_image = image.crop((0, 0, img_width, img_height - 20))
            # save the image
            cropped_image.save(image)
        except Exception as e:
            print("Error cropping the bottom off of the image: " + str(e))


def check_colors_of_six_panels(image):
    # there are six panels in the image (side by side) and we want to check the colors of each panel
    # get the image size
    # the input image should be an image object

    img_width, img_height = get_image_size(image)
    # get the width of each panel
    panel_width = img_width / 6
    # get the height of each panel (remove botttom 20 pixels)
    panel_height = img_height - 20
    # get the colors of each panel
    panel_1 = image.getpixel((panel_width / 2, panel_height / 2))
    panel_2 = image.getpixel((panel_width * 1.5, panel_height / 2))
    panel_3 = image.getpixel((panel_width * 2.5, panel_height / 2))
    panel_4 = image.getpixel((panel_width * 3.5, panel_height / 2))
    panel_5 = image.getpixel((panel_width * 4.5, panel_height / 2))
    panel_6 = image.getpixel((panel_width * 5.5, panel_height / 2))
    # return the number of panels that meet the color criteria:
    # criteria:
    # red = less than 250 and greater than 170
    # green = less than 250 and greater than 170
    # blue = less than 250 and greater than 170
    # orange = less than 250 and greater than 170

    # check panel 1
    # if panel_1[0] < 250 and panel_1[0] > 170 and panel_1[1] < 250 and panel_1[1] > 170 and panel_1[2] < 250 and panel_1[2] > 170:
    #     panel_1_color = 'white'
    # elif panel_1[0] < 250 and panel_1[0] > 170 and panel_1[1] < 250 and panel_1[1] > 170 and panel_1[2] < 170:
    #     panel_1_color = 'yellow'
    # elif panel_1[0] < 250 and panel_1[0] > 170 and panel_1[1] < 170 and panel_1[2] < 250 and panel_1[2] > 170:
    #     panel_1_color = 'green'
    # elif panel_1[0] < 170 and panel_1[1] < 250 and panel_1[1] > 170 and panel_1[2] < 250 and panel_1[2] > 170:
    #     panel_1_color = 'blue'
    # elif panel_1[0] < 250 and panel_1[0] > 170 and panel_1[1] < 170 and panel_1[2] < 170:
    #     panel_1_color = 'red'
    # else:
    #     panel_1_color = 'other'
    # this is an interesting code but let's do something simpler, I want to isolate images that have sunsets in them, so let's just check the red and blue values
    if (
        panel_1[0] > 200 and panel_1[2] > 200
    ):  # if the red and blue values are greater than 200, then it's a sunset?
        panel_1_result = True  # set the result to true
    else:
        panel_1_result = False
    # check panel 2
    if panel_2[0] > 200 and panel_2[2] > 200:
        panel_2_result = True
    else:
        panel_2_result = False
    # check panel 3
    if panel_3[0] > 200 and panel_3[2] > 200:
        panel_3_result = True
    # check panel 4
    if panel_4[0] > 200 and panel_4[2] > 200:
        panel_4_result = True
    # check panel 5
    if panel_5[0] > 200 and panel_5[2] > 200:
        panel_5_result = True
    # check panel 6
    if panel_6[0] > 200 and panel_6[2] > 200:
        panel_6_result = True
    # return the results
    panels_collection = [
        panel_1,
        panel_2,
        panel_3,
        panel_4,
        panel_5,
        panel_6,
    ]  # put the panels into a list
    return (
        panel_1,
        panel_2,
        panel_3,
        panel_4,
        panel_5,
        panel_6,
        panel_1_result,
        panel_2_result,
        panel_3_result,
        panel_4_result,
        panel_5_result,
        panel_6_result,
    )


def get_panel_segments(image):
    """
    get_panel_segments takes an image and returns the segments of the image that are the panels

    :param image: the image to be segmented

    :param image: the image to be segmented
    :type image: image object (from PIL) or numpy array (from OpenCV)
    :return: the segments of the image that are the panels
    :rtype: list of image objects
    """
    # get the image size
    img_width, img_height = image.shape[0], image.shape[1]
    # get the width of each panel
    panel_width = img_width / 6
    # get the height of each panel (remove botttom 20 pixels)
    panel_height = img_height - 20
    # cast them to integers
    panel_width = int(panel_width)
    panel_height = int(panel_height)

    # get the segments of the image that are the panels
    panel_1 = image[
        0:panel_height, 0:panel_width
    ]  # to overcome the error: slice indices must be integers or None or have an __index__ method
    # we have to convert the panel_1 to an image object
    panel_2 = image[0:panel_height, panel_width : panel_width * 2]
    panel_3 = image[0:panel_height, panel_width * 2 : panel_width * 3]
    panel_4 = image[0:panel_height, panel_width * 3 : panel_width * 4]
    panel_5 = image[0:panel_height, panel_width * 4 : panel_width * 5]
    panel_6 = image[0:panel_height, panel_width * 5 : panel_width * 6]
    # return the segments
    panels = [panel_1, panel_2, panel_3, panel_4, panel_5, panel_6]
    return panels


def get_average_color(image):
    """
    get_average_color takes an image and returns the average color of the image

    :param image: the image to be segmented
    :type image: image object (from PIL) or numpy array (from OpenCV)
    :return: the average color of the image
    :rtype: tuple of integers
    """
    # get the image size
    img_width, img_height = image.shape[0], image.shape[1]
    # get the average color of the image
    # to do this we have to convert the image from a numpy array to a PIL image
    image = Image.fromarray(image)
    average_color = image.getpixel((img_width // 2, img_height // 2))
    # return the average color
    return average_color


def generate_one_full_day(buoy_id, day_number, month_number, year_number):
    # concatenate one image for each hour of the day and save it as a new image called 'full_day.png'
    # get the images from the buoy_id folder and sort them
    images = os.listdir(buoy_id)
    images.sort()
    # the names of the images are in the format: '2019-01-01_00.png'
    # images/buoys/41004/2022_11_5_15_44.jpg
    # where the day is the 3rd element in the list
    # and the hour is the 4th element in the list
    # so we can use a list comprehension to get the images for each hour of the day if we sort the images first.
    # go through the folder and add the filenames to the list if the day is the same as the day_number
    # and the month is the same as the month_number
    # and the year is the same as the year_number

    # get the images for the day
    images_for_day = [
        image
        for image in images
        if image.split("_")[2] == str(day_number)
        and image.split("_")[1] == str(month_number)
        and image.split("_")[0] == str(year_number)
    ]
    # get the images for the day and sort them
    images_for_day.sort()
    # get one image for each hour of the day
    images_for_day = [
        image
        for image in images_for_day
        if image.split("_")[3] == "00"
        or image.split("_")[3] == "01"
        or image.split("_")[3] == "02"
        or image.split("_")[3] == "03"
        or image.split("_")[3] == "04"
        or image.split("_")[3] == "05"
        or image.split("_")[3] == "06"
        or image.split("_")[3] == "07"
        or image.split("_")[3] == "08"
        or image.split("_")[3] == "09"
        or image.split("_")[3] == "10"
        or image.split("_")[3] == "11"
        or image.split("_")[3] == "12"
        or image.split("_")[3] == "13"
        or image.split("_")[3] == "14"
        or image.split("_")[3] == "15"
        or image.split("_")[3] == "16"
        or image.split("_")[3] == "17"
        or image.split("_")[3] == "18"
        or image.split("_")[3] == "19"
        or image.split("_")[3] == "20"
        or image.split("_")[3] == "21"
        or image.split("_")[3] == "22"
        or image.split("_")[3] == "23"
    ]
    # get the images for the day and sort them
    images_for_day.sort()
    # add the images one after the other by concatenating them vertically and save the result as a new image called 'full_day.png' in the buoy_id folder
    # get the first image
    first_image = images_for_day[0]
    # open the first image
    first_image = Image.open(buoy_id + "/" + first_image)
    # get the width and height of the first image
    width, height = first_image.size
    # create a new image with the same width as the first image and the height of the first image multiplied by the number of images
    new_image = Image.new("RGB", (width, height * len(images_for_day)))
    # paste the first image in the new image
    new_image.paste(first_image, (0, 0))
    # go through the rest of the images and add them to the bottom of the new image one after the other
    for i in range(1, len(images_for_day)):
        # get the image
        image = images_for_day[i]
        # open the image
        image = Image.open(buoy_id + "/" + image)
        # paste the image in the new image
        new_image.paste(image, (0, height * i))
    # save the new image
    new_image.save(buoy_id + "/full_day.png")


### Testing To find Red


def finding_red_version_two(image):
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image_height = image.shape[0]
    image_width = image.shape[1]
    # https://stackoverflow.com/questions/30331944/finding-red-color-in-image-using-python-opencv
    image_result = np.zeros((image_height, image_width, 3), np.uint8)
    for i in range(image_height):  # those are set elsewhere
        for j in range(image_width):  # those are set elsewhere
            if (
                img_hsv[i][j][1] >= 50
                and img_hsv[i][j][2] >= 50
                and (img_hsv[i][j][0] <= 10 or img_hsv[i][j][0] >= 170)
            ):
                image_result[i][j] = img_hsv[i][
                    j
                ]  # this is the red (above is saturation, value, and hue)
    return image_result


def finding_red_version_three(image_path):
    """
    finding_red_version_three takes an image and returns the red pixels in the image

    :param image_path: the path to the image to be segmented

    :param image_path: the path to the image to be segmented
    :type image_path: string
    :return: the red pixels in the image
    :rtype: image object
    """
    img = cv2.imread(image_path)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # lower mask (0-10)
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])
    mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

    # upper mask (170-180)
    lower_red = np.array([170, 50, 50])
    upper_red = np.array([180, 255, 255])
    mask1 = cv2.inRange(img_hsv, lower_red, upper_red)

    # join my masks
    mask = mask0 + mask1

    # set my output img to zero everywhere except my mask
    # output_img = img.copy()
    # output_img[np.where(mask==0)] = 0

    # or your HSV image, which I *believe* is what you want
    output_hsv = img_hsv.copy()
    output_hsv[np.where(mask == 0)] = 0

    # save the output_img to 'test.png' in images folder
    # cv2.imwrite('images/test.png', output_img)
    # save the output_hsv to 'test_hsv.png' in images folder
    cv2.imwrite("images/test_hsv.png", output_hsv)
    # why is the output_hsv image all black?
    # because the output_hsv image is in HSV format and not RGB format
    # so we have to convert it to RGB format
    # https://stackoverflow.com/questions/15007348/convert-hsv-to-rgb-using-python-and-opencv
    # converting the image from HSV to RGB
    output_hsv = cv2.cvtColor(output_hsv, cv2.COLOR_HSV2RGB)
    # save the output_hsv to 'test_hsv.png' in images folder
    cv2.imwrite("images/test_hsv.png", output_hsv)


def detect_red_v4(image):
    # Red color
    if type(image) == str:
        image = cv2.imread(image)  # read the image
    low_red = np.array([161, 155, 84])
    high_red = np.array([179, 255, 255])
    red_mask = cv2.inRange(image, low_red, high_red)
    percent_pixels_red = np.sum(red_mask) / (image.shape[0] * image.shape[1])
    return percent_pixels_red


def detect_red(img):
    """
    detect_red _summary_

    _extended_summary_

    :param image: _description_
    :type image:
    :return: _description_
    :rtype: _type_
    """
    try:
        # hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # creating a mask to catch the color red in the image
        # Here, you define your target color as
        # a tuple of three values: RGB
        # red = [130, 158, 0]
        red = [0, 0, 255]  # this is the color of the red in the image

        # You define an interval that covers the values
        # in the tuple and are below and above them by 20
        diff = 20

        # Be aware that opencv loads image in BGR format,
        # that's why the color values have been adjusted here:
        boundaries = [
            (
                [red[2], red[1] - diff, red[0] - diff],
                [red[2] + diff, red[1] + diff, red[0] + diff],
            )
        ]

        # Scale your BIG image into a small one:
        scalePercent = 0.3

        # Calculate the new dimensions
        width = int(img.shape[1] * scalePercent)
        height = int(img.shape[0] * scalePercent)
        newSize = (width, height)

        # Resize the image:
        img = cv2.resize(img, newSize, None, None, None, cv2.INTER_AREA)

        # check out the image resized:
        #!cv2.imshow("img resized", img)
        #!cv2.waitKey(0)

        # for each range in your boundary list:
        for (lower, upper) in boundaries:

            # You get the lower and upper part of the interval:
            lower = np.array(lower, dtype=np.uint8)
            upper = np.array(upper, dtype=np.uint8)

            # cv2.inRange is used to binarize (i.e., render in white/black) an image
            # All the pixels that fall inside your interval [lower, uipper] will be white
            # All the pixels that do not fall inside this interval will
            # be rendered in black, for all three channels:
            mask = cv2.inRange(img, lower, upper)

            # Check out the binary mask:
            #!cv2.imshow("binary mask", mask)
            # cv2.waitKey(0)

            # Now, you AND the mask and the input image
            # All the pixels that are white in the mask will
            # survive the AND operation, all the black pixels
            # will remain black
            output = cv2.bitwise_and(img, img, mask=mask)

            # Check out the ANDed mask:
            #!cv2.imshow("ANDed mask", output)
            # cv2.waitKey(0)

            # You can use the mask to count the number of white pixels.
            # Remember that the white pixels in the mask are those that
            # fall in your defined range, that is, every white pixel corresponds
            # to a red pixel. Divide by the image size and you got the
            # percentage of red pixels in the original image:
            ratio_red = cv2.countNonZero(mask) / (img.size / 3)

            # This is the color percent calculation, considering the resize I did earlier.
            colorPercent = (ratio_red * 100) / scalePercent

            # Print the color percent, use 2 figures past the decimal point
            print("red pixel percentage:", np.round(colorPercent, 2))

            # numpy's hstack is used to stack two images horizontally,
            # so you see the various images generated in one figure:
            #!cv2.imshow("images", np.hstack([img, output]))
            # save the image as 'test.png' in the images folder
            cv2.imwrite("images/test.png", np.hstack([img, output]))
            # cv2.waitKey(0)
    except Exception as e:
        print("Error in detect_red: ", e)


def deal_with_white_images_and_populate_tapestry():
    sunsets_found = 0  # keep track of how many sunsets we find
    files = glob.glob("images/buoys/*/*")
    # without glob
    # files = []
    # for file in os.listdir('images/buoys/'):
    #   files.append('images/buoys/' + file)
    # print(files)
    height, width, channels = cv2.imread(files[0]).shape
    # blank_image = np.zeros((height*10, width, channels), np.uint8)
    # get the ten images that have the most orange in them
    # make the blank image the same size as the images

    # shuffle the files so we don't always get the same ten images
    #!random.shuffle(files) #note: this could be a problem later

    add_list = []

    for file in tqdm(files):
        # read the image
        try:
            image = cv2.imread(file)
            if not is_recent(file, 300):  # 300 minutes
                continue

            if panel_mode:
                # get the image details for panels 1-6
                (
                    panel_1,
                    panel_2,
                    panel_3,
                    panel_4,
                    panel_5,
                    panel_6,
                ) = get_panel_segments(image)
                # explanation of results:
                panels_collection = [
                    panel_1,
                    panel_2,
                    panel_3,
                    panel_4,
                    panel_5,
                    panel_6,
                ]  # put the panels into a list

                # put True into panel_results once for each panel (1/6 of the width of the image) that has an average red value greater than 180 and an average blue value greater than 180

                # the image passed to get_panel_segments should be a cv2 image
                assert type(image) == np.ndarray

                panel_segments = get_panel_segments(image)

                positive_panel_count = (
                    0  # keep track of how many panels have a sunset in them
                )
                # get the average color of each panel
                for panel in panel_segments:
                    panel_average_color = get_average_color(panel)
                    # check if the panel is a sunset
                    if panel_average_color[0] > 200 and panel_average_color[2] > 200:
                        # increment the positive_panel_count by 1
                        positive_panel_count += 1

                # now check if the positive_panel_count is greater than 3 (i.e. more than half of the panels have a sunset in them)
                if positive_panel_count > 3:
                    add_list.append(file)
                    sunsets_found += 1
                else:
                    continue  # if the positive_panel_count is not greater than 3, then continue on to the next image

                # what is the average amount of red in the image?
                # what is the average amount of blue in the image?
                # what is the average amount of green in the image?
                # what is the average amount of orange in the image?

                # x = [[panel_1_result, panel_2_result, panel_3_result, panel_4_result, panel_5_result, panel_6_result],panels_collection] # return the results
                # panel_results - a list of true or false values for each panel (true if the panel is orange, false if not).
                # panels_collection - a list of the colors of each panel (in RGB format) (this is for debugging purposes)
                # if the image has at least 4 panels that are orange, then we want to add it to the tapestry

                # note: uncomment below if the check_colors_of_six_panels function is not working
                # # get the average orange value
                # orange_value = np.mean(image[:,:,2]) # the orange channel is the third channel
                # red_value = np.mean(image[:,:,0]) # get the average red value when over 147 then save the image

                # get the median orange value across the panels
                panels_collection = panel_segments  # put the panels into a list
                orange_value = np.median(
                    [
                        panels_collection[0][2],
                        panels_collection[1][2],
                        panels_collection[2][2],
                        panels_collection[3][2],
                        panels_collection[4][2],
                        panels_collection[5][2],
                    ]
                )
                # # get the median red value across the panels
                red_value = np.median(
                    [
                        panels_collection[0][0],
                        panels_collection[1][0],
                        panels_collection[2][0],
                        panels_collection[3][0],
                        panels_collection[4][0],
                        panels_collection[5][0],
                    ]
                )

            # if the average amount of orange is greater than 200, then add the image to the add_list

            # if the average amount of orange is greater than 200:
            # add_list.append(file)
            # sunsets_found += 1
            # else:
            # continue # if the average amount of orange is not greater than 200, then continue on to the next image

            # * just check the image to see if red is less than 20, green is less than 20, and blue is less than 20
            # * if so then skip the image

            red_score = np.mean(image[:, :, 0])
            green_score = np.mean(image[:, :, 1])
            blue_score = np.mean(image[:, :, 2])
            if red_score < 20 and green_score < 20 and blue_score < 20:
                # print('Night image detected')
                continue
            else:
                # print('Day image detected')
                red_val = detect_red_v4(image)
                if red_val > 2.5:
                    print(
                        " ---- > Sunset detected? ", red_val
                    )  # print the sunset detected message
                    # save the image to the sunset folder under the appropriate buoy
                    buoy_name = file.split("/")[2]
                    buoy_folder = "images/sunsets/" + buoy_name
                    if not os.path.exists(buoy_folder):
                        os.makedirs(buoy_folder)
                    cv2.imwrite(buoy_folder + "/" + file.split("/")[3], image)
                    red_flag = True
                elif red_val > 15:
                    print(
                        " ---- > super sunset detected? ", red_val
                    )  # print the sunset detected message
                    # save the image to the keepers folder under the appropriate buoy
                    buoy_name = file.split("/")[2]
                    buoy_folder = "images/keepers/" + buoy_name
                    if not os.path.exists(buoy_folder):
                        os.makedirs(buoy_folder)
                    cv2.imwrite(buoy_folder + "/" + file.split("/")[3], image)
                    red_flag = True
                else:
                    red_flag = False
                if red_flag:  # if the image has more than 10% red in it
                    add_list.append(file)
                    sunsets_found += 1

            # blue_value = np.mean(image[:,:,1]) # blue value
            # print(orange_value, red_value, blue_value)
            # show the image and annotate it with the orange, red, and blue values
            # plt.imshow(image)
            # plt.title("Orange: " + str(orange_value) + " Red: " + str(red_value) + " Blue: " + str(blue_value))
            # plt.show()

            # save the filename to a list if the image is to be added to the tapestry

        except Exception as e:
            print(e)
            continue

        blank_image = np.zeros((height * len(add_list), width, channels), np.uint8)
        try:
            cv2.imwrite("images/tapestry.png", blank_image)
        except Exception as e:
            print(e)
            print("Could not write blank image")
            print("line 322")
            continue

    for file in tqdm(add_list):
        # read the image
        try:
            image = cv2.imread(file)
            # get the average orange value
            # print(np.mean(image[:,:,2]))
            # orange_value = np.mean(image[:,:,2]) # the orange channel is the third channel

            # red_value = np.mean(image[:,:,0]) # get the average red value when over 147 then save the image

            # daytime images always have higher values than 10 for all three channels
            # values less than 10 are usually night
            # skip the image if it is night
            # if orange_value < 10 and red_value < 10: # higher than 250 for all means it is a white imag
            #     continue
            # # if the values are all higher than 250 then it is a white image and we want to remove it
            # if orange_value > 250 and red_value > 250:
            #     os.remove(file)
            #     print("Removed white image")
            #     continue
            # # if the image was not taken in the last x hours, skip it
            # if not is_recent(file, 60): # 60 minutes
            #     continue

            # blue_value = np.mean(image[:,:,1]) # blue value
            # print(orange_value, red_value, blue_value)
            # show the image and annotate it with the orange, red, and blue values
            # plt.imshow(image)
            # plt.title("Orange: " + str(orange_value) + " Red: " + str(red_value) + " Blue: " + str(blue_value))
            # plt.show()

            # if we reached this point the image can be added to the tapestry unless the tapestry has already been filled then just keep going without adding the image
            if (
                rotating
            ):  # if the tapestry is rotating, we take an image and add it to the tapestry as well as remove the oldest image otherwise we just add the image to the tapestry
                if sunsets_found == 10:
                    # remove the top image from the tapestry
                    # get the image at the top of the tapestry which has a height of total_height/10
                    top_image = blank_image[0:height, 0:width]
                    # crop the image to remove the top 1/10th of the image
                    blank_image = blank_image[height : height * 10, 0:width]
                    # add the new image to the bottom of the tapestry
                    blank_image = np.concatenate((blank_image, image), axis=0)
                    cv2.imwrite("images/tapestry.png", blank_image)
                else:
                    blank_image[
                        sunsets_found * height : (sunsets_found + 1) * height, 0:width
                    ] = image
                    # show progress by printing out the blank image
                    cv2.imwrite("images/tapestry.png", blank_image)
                    # print("Sunset found!")
                    sunsets_found += 1  # increment the number of sunsets found

            else:
                blank_image[
                    sunsets_found * height : (sunsets_found + 1) * height, 0:width
                ] = image
                # show progress by printing out the blank image
                cv2.imwrite("images/tapestry.png", blank_image)
                # print("Sunset found!")
                sunsets_found += 1  # increment the number of sunsets found
        except:
            print("Error reading image")
            print("line 386")
            pass


def stitched_panoramas(panel1, panel2, panel3, panel4, panel5, panel6):
    # get the image size
    img_width, img_height = panel1.shape[1], panel1.shape[0]
    # get the ratio of the width to height
    r = float(img_width) / float(img_height)
    # get the aspect ratio of the image
    ar = round(r, 2)
    # calculate the rotation angle
    rot = math.degrees(math.atan2(-panel1.get_top(), -panel1.get_left()))
    # get the rotation matrix for this angle
    m = cv2.getRotationMatrix2D((img_width / 2, img_height / 2), rot, 1)
    # multiply the two matrices together to get the final transformation
    new_im = cv2.warpAffine(panel1, m, (img_width, img_height))
    # find the top-left corner of the image
    pos = np.array((new_im.shape.width / 2, new_im.shape.height / 2))
    # crop the image to the correct size
    new_im = new_im.copy()
    #!cropped_im = new_im.crop(pos)
    # rotate the image back
    rotated_im = cv2.rotate(new_im, 90, pos, 0, 0, 360)
    # resize the image to the right size
    resized_im = cv2.resize(
        rotated_im, (int(round(ar * img_width)), int(round(ar * img_height)))
    )
    return resized_im


def get_image_size(image):
    """
    get_image_size returns the width and height of an image

    _extended_summary_

    :param image: the image to get the size of
    :type image: cv2 image
    :return: the width and height of the image
    :rtype: tuple
    """
    # get the image width and height
    w, h = image.shape[:2]
    # I am getting Exception has occurred: ValueError
    # too many values to unpack (expected 2)
    # the way to fix this is...
    # w, h = image.shape[:2]

    # get the ratio of the width to height
    r = float(w) / float(h)
    # get the aspect ratio of the image
    ar = round(r, 2)
    # calculate the rotation angle
    rot = math.degrees(math.atan2(-image.get_top(), -image.get_left()))
    # get the rotation matrix for this angle
    m = cv2.getRotationMatrix2D((w / 2, h / 2), rot, 1)
    # multiply the two matrices together to get the final transformation
    new_im = cv2.warpAffine(image, m, (w, h))
    # find the top-left corner of the image
    pos = np.array((new_im.shape.width / 2, new_im.shape.height / 2))
    # crop the image to the correct size
    new_im = new_im.crop(pos)
    # rotate the image back
    rotated_im = cv2.rotate(new_im, 90, pos, 0, 0, 360)
    # resize the image to the right size
    resized_im = cv2.resize(rotated_im, (int(round(ar * w)), int(round(ar * h))))
    return resized_im


# Implementing the trained h5 model in real-time using OpenCV and Python to detect scene elements or types in panel images from the cameras on the buoys.
# The model is already trained and saved in the models/buoy_model/keras_model.h5 file.

# # import the necessary packages
# from imutils import paths
# import numpy as np
# import cv2

# # import the load_model function from keras
# from keras.models import load_model

# # load the model from disk
# print("[INFO] loading model...")
# model = load_model('models/buoy_model/keras_model.h5')
# Labels for this model are shown below
# 0 Direct Sun
# 1 Stormy Weather
# 2 Interesting
# 3 Object Detected
# 4 Sunset
# 5 Clouds
# 6 Night


# def check_image(image):
#     global model # load the model from disk
#     # load the image and show it
#     image = cv2.imread('images/panels/44020/2022_11_6_10_54/panel_1.jpg')
#     cv2.imshow("image", image)
#     cv2.waitKey(0) # show image
#     # feed the image to the model and get the prediction
#     prediction = model.predict(image) # predict the image
#     print(prediction) # print the prediction

#     # Return the predicted class of the image (0-6)
#     return np.argmax(prediction) # return the predicted class i.e. the type of scene in the image 'Stormy Weather' or 'Direct Sun' etc.
#     # We return the np.argmax of the prediction because the prediction is a list of probabilities for each class. The class with the highest probability is the predicted class.


# # Do a test run of the check_image function to see if it works on a single image
# test_image = 'images/panels/42002/2022_11_6_18_30/panel_3.png'
# validation_class = 'Sunset' # we know that this image is a sunset image, so we can use this to test the model
# # we can use this to test the model

# # Testing Model:
# print("Testing Keras Model")
# assert(os.path.exists(test_image)) # check if file exists
# pred = check_image(test_image) # get the predicted class of the image
# if pred != 4:
#     print("The image is not a sunset image")
#     print(pred) # print the predicted class
# else:
#     print("The image is a sunset image")


def buoy_links():
    global ids
    links = [
        "https://www.ndbc.noaa.gov/buoycam.php?station=42001",
        "https://www.ndbc.noaa.gov/buoycam.php?station=46059",
        "https://www.ndbc.noaa.gov/buoycam.php?station=41044",
        "https://www.ndbc.noaa.gov/buoycam.php?station=46071",
        "https://www.ndbc.noaa.gov/buoycam.php?station=42002",
        "https://www.ndbc.noaa.gov/buoycam.php?station=46072",
        "https://www.ndbc.noaa.gov/buoycam.php?station=46066",
        "https://www.ndbc.noaa.gov/buoycam.php?station=41046",
        "https://www.ndbc.noaa.gov/buoycam.php?station=46088",
        "https://www.ndbc.noaa.gov/buoycam.php?station=44066",
        "https://www.ndbc.noaa.gov/buoycam.php?station=46089",
        "https://www.ndbc.noaa.gov/buoycam.php?station=41043",
        "https://www.ndbc.noaa.gov/buoycam.php?station=42012",
        "https://www.ndbc.noaa.gov/buoycam.php?station=42039",
        "https://www.ndbc.noaa.gov/buoycam.php?station=46012",
        "https://www.ndbc.noaa.gov/buoycam.php?station=46011",
        "https://www.ndbc.noaa.gov/buoycam.php?station=42060",
        "https://www.ndbc.noaa.gov/buoycam.php?station=41009",
        "https://www.ndbc.noaa.gov/buoycam.php?station=46028",
        "https://www.ndbc.noaa.gov/buoycam.php?station=44011",
        "https://www.ndbc.noaa.gov/buoycam.php?station=41008",
        "https://www.ndbc.noaa.gov/buoycam.php?station=46015",
        "https://www.ndbc.noaa.gov/buoycam.php?station=42059",
        "https://www.ndbc.noaa.gov/buoycam.php?station=44013",
        "https://www.ndbc.noaa.gov/buoycam.php?station=44007",
        "https://www.ndbc.noaa.gov/buoycam.php?station=46002",
        "https://www.ndbc.noaa.gov/buoycam.php?station=51003",
        "https://www.ndbc.noaa.gov/buoycam.php?station=46027",
        "https://www.ndbc.noaa.gov/buoycam.php?station=46026",
        "https://www.ndbc.noaa.gov/buoycam.php?station=51002",
        "https://www.ndbc.noaa.gov/buoycam.php?station=51000",
        "https://www.ndbc.noaa.gov/buoycam.php?station=42040",
        "https://www.ndbc.noaa.gov/buoycam.php?station=44020",
        "https://www.ndbc.noaa.gov/buoycam.php?station=46025",
        "https://www.ndbc.noaa.gov/buoycam.php?station=41010",
        "https://www.ndbc.noaa.gov/buoycam.php?station=41004",
        "https://www.ndbc.noaa.gov/buoycam.php?station=51001",
        "https://www.ndbc.noaa.gov/buoycam.php?station=44025",
        "https://www.ndbc.noaa.gov/buoycam.php?station=41001",
        "https://www.ndbc.noaa.gov/buoycam.php?station=51004",
        "https://www.ndbc.noaa.gov/buoycam.php?station=44027",
        "https://www.ndbc.noaa.gov/buoycam.php?station=41002",
        "https://www.ndbc.noaa.gov/buoycam.php?station=42020",
        "https://www.ndbc.noaa.gov/buoycam.php?station=46078",
        "https://www.ndbc.noaa.gov/buoycam.php?station=46087",
        "https://www.ndbc.noaa.gov/buoycam.php?station=51101",
        "https://www.ndbc.noaa.gov/buoycam.php?station=46086",
        "https://www.ndbc.noaa.gov/buoycam.php?station=45002",
        "https://www.ndbc.noaa.gov/buoycam.php?station=46053",
        "https://www.ndbc.noaa.gov/buoycam.php?station=46047",
        "https://www.ndbc.noaa.gov/buoycam.php?station=46084",
        "https://www.ndbc.noaa.gov/buoycam.php?station=46085",
        "https://www.ndbc.noaa.gov/buoycam.php?station=45003",
        "https://www.ndbc.noaa.gov/buoycam.php?station=45007",
        "https://www.ndbc.noaa.gov/buoycam.php?station=46042",
        "https://www.ndbc.noaa.gov/buoycam.php?station=45012",
        "https://www.ndbc.noaa.gov/buoycam.php?station=42019",
        "https://www.ndbc.noaa.gov/buoycam.php?station=46069",
        "https://www.ndbc.noaa.gov/buoycam.php?station=46054",
        "https://www.ndbc.noaa.gov/buoycam.php?station=41049",
        "https://www.ndbc.noaa.gov/buoycam.php?station=45005",
    ]

    # note: undo this to go with the established buoy list
    # links_2 = create_buoy_links(ids)

    # # append the links_2 to links if they are not already in links
    # for link in links_2:
    #     if link not in links:
    #         links.append(link)

    return links


# ids from cam_backend


def create_buoy_links(ids):
    # for each id in ids, create a link
    links = []
    for id in ids:
        link = "https://www.ndbc.noaa.gov/buoycam.php?station=" + id
        links.append(link)
    return links


# Notes:
# Buoy 42002 Has good sunsets


def check_buoy_image_ifwhite(image):
    """
    check_buoy_image_ifwhite checks if the image is white

    This function checks if the image is white. If the image is white, then the image is not valid and should be deleted.

    :param image: the image to check
    :type image: result of requests library get request for image url
    :return: True if the image is white, False if the image is not white
    :rtype: bool
    """
    # some buoys do not have a camera or the camera is not working. In these cases the image is white with only the text "No Image Available"
    # determine if the image is white
    # get the image in numpy array format
    img = np.asarray(bytearray(image.content), dtype="uint8")
    # convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # check if the image is white
    if np.mean(gray) > 250:
        return True
    else:
        return False


def ocean_stitching(imagePaths, pano_path):
    images = []
    # loop over the image paths, load each one, and add them to our
    # images to stitch list
    # open each image with cv2 and append to images list
    for imagePath in imagePaths:
        try:
            # add the full path to the image
            # '/Volumes/Backups of Grahams IMAC/PythonProjects/PySeas_Master_Folder/PySeas/images/panels/44020/2022_11_6_10_54/panel_1.jpg'
            # read the image
            full_path = (
                "/Volumes/Backups of Grahams IMAC/PythonProjects/PySeas_Master_Folder/PySeas/"
                + imagePath
            )
            imagePath = full_path  # this is the full path to the image
            assert os.path.exists(imagePath)  # check if file exists
            image = cv2.imread(imagePath, cv2.IMREAD_COLOR)  # read image
            # cv2.imshow("image", image)
            # cv2.waitKey(0) # show image
            images.append(image)  # append to list
            if image is None:
                print("Error loading image: " + imagePath)
        except:
            print("Error reading image: " + imagePath)
            continue
    # getting error: OpenCV(4.6.0) /Users/runner/work/opencv-python/opencv-python/opencv/modules/imgproc/src/resize.cpp:4052: error: (-215:Assertion failed) !ssize.empty() in function 'resize'
    # how to fix this:
    # The error is saying that the image is empty. This is because you are trying to read an image that doesn't exist. Check the path to the image and make sure it is correct.

    # initialize OpenCV's image stitcher object and then perform the image
    # stitching
    print("[INFO] stitching images...")
    stitcher = cv2.createStitcher() if imutils.is_cv3() else cv2.Stitcher_create()
    (status, stitched) = stitcher.stitch(images)
    # if the status is '0', then OpenCV successfully performed image
    # stitching
    if status == 0:
        # write the output stitched image to disk
        cv2.imwrite(pano_path, stitched)  # save the stitched image
        # display the output stitched image to our screen
        #!cv2.imshow("Stitched", stitched)
        #!cv2.waitKey(0)
    # otherwise the stitching failed, likely due to not enough keypoints)
    # being detected
    else:
        print("[INFO] image stitching failed ({})".format(status))


def refine_view(stitched_image):
    stitched = cv2.copyMakeBorder(
        stitched, 10, 10, 10, 10, cv2.BORDER_CONSTANT, (0, 0, 0)
    )
    # convert the stitched image to grayscale and threshold it
    # such that all pixels greater than zero are set to 255
    # (foreground) while all others remain 0 (background)
    gray = cv2.cvtColor(stitched, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]
    # find all external contours in the threshold image then find
    # the *largest* contour which will be the contour/outline of
    # the stitched image
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    # allocate memory for the mask which will contain the
    # rectangular bounding box of the stitched image region
    mask = np.zeros(thresh.shape, dtype="uint8")
    (x, y, w, h) = cv2.boundingRect(c)
    cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)


# given a list of images, create a pano image
def create_pano_image(image_list, pano_path):
    # create a pano image from the images in image_list
    # image_list is a list of image paths
    # pano_path is the path to save the pano image
    # create the pano image
    ocean_stitching(image_list, pano_path)
    # refine the pano image
    # refine_view(pano_path)


def chunk_images(buoy_id, foldername):
    """
    chunk_images takes a folder of images and splits them into sets of 6 images

    _extended_summary_

    :param buoy_id: The id of the buoy
    :type buoy_id: int
    :param foldername: The name of the folder containing the images
    :type foldername: str
    #"""
    # #buoy_id = str(foldername)
    # # There are many images in the folder. We want to parse these in sets of six.
    # # images/panels/41002/2022_11_5_15_44_panel_1.png is an example of the naming convention for the images.
    # # the first number (folder under panels) is the buoy id, then the filename begins with the year, the second number is the month, the third number is the day, the fourth number is the hour, the fifth number is the minute, and the sixth number is the panel number.
    # # we want to select the images that have the same buoy id, year, month, day, and hour AND end in the panel numbers: 1,2,3,4,5,6.

    # # for the example file above 2022_11_5_15_44_panel_1.png
    # # we want to put it into a subfolder named 2022_11_5_15 and put all the images that match the naming convention into that folder. There should only be 6.
    # # then return the list of lists of images that are in the subfolders.

    # # if the name of the

    # # foldername[0] --> '2022_11_5_15_44.jpg_panel_1.png'
    # # this is the kind of dirty names we have to clean.

    # # we want to get the first 4 numbers and the last number.
    # # we want to put the images into a folder named 2022_11_5_15
    # # we want to return a list of lists of images that are in the subfolders.

    # # we want to get the first 4 numbers and the last number.
    # # if type(foldername) == list:
    # #     foldername = foldername[0]
    # # else:
    # #     foldername = foldername
    # for file_name in tqdm(foldername):

    #     first_four = file_name.split('_')[0:4] # i.e. ['2022', '11', '5', '15']
    #     last_num = file_name.split('_')[-1] # i.e. '1.png'
    #     last_num = last_num.split('.')[0] # i.e. '1'

    #     # we want to put the images into a folder named 2022_11_5_15
    #     new_folder_name = '_'.join(first_four) # i.e. '2022_11_5_15'

    #     # the photos that belong with the new folder name (2022_11_5_15) contain the foldername's numbers.
    #     # Sort the images in the folder by the date they were added to the folder.
    #     #sorted_foldername = sorted(foldername, key=lambda x: os.path.getmtime(x)) # sort the images by the date they were added to the folder.

    #     # find all the files with the first 4 numbers (2022_11_5_15) and .jpg only once in the name
    #     files_with_last_four = [x for x in foldername if new_folder_name in x and 'jpg_panel' not in x]
    #     # find all the files with the last number (1)
    #     files_with_last_num = [x for x in files_with_last_four if str('panel_')+last_num in x]

    #     # are the images in the foldername in the correct order?
    #     # we want to check if the images are in the correct order.
    #     # They should now have the word panel 1, in the first image, panel 2 in the second image, etc. down to panel 6 and then the next image should be panel 1 again.
    #     # now we can just take each six images and put them into a folder.

    #     # logic
    #     # as long as the image contains "panel_n" where n is the next number in the order of the panels we add the image to the folder name that matches the first four numbers in their name.

    #      # go through the images in the foldername and put them into the correct folder.
    #     # we want to put the images into a folder named 2022_11_5_15 or whatever the first four numbers are.

    #     # make a folder in the panels folder with the name of the first four numbers.
    #     # if the folder doesn't exist, make it.
    #     print('new_folder_name',new_folder_name)
    #     try:
    #         os.mkdir(f'images/panels/{buoy_id}/batches/{new_folder_name}')
    #     except Exception as e:
    #         print(e)
    #         pass

    #     # go through the images in the foldername and put them into the correct folder.
    #     # we want to put the images into a folder named 2022_11_5_15 or whatever the first four numbers are.
    #     for image in files_with_last_num:
    #         # move the image into the directory f'images/panels/{buoy_id}/{new_folder_name}'
    #         # full path of the image
    #         full_path = f'images/panels/{buoy_id}/{image}'
    #         shutil.move(full_path, f'images/panels/{buoy_id}/batches/{new_folder_name}')
    #         pass

    #     # now we want to delete the images that are in the foldername but not in the subfolder.
    #     # we want to delete the images that are in the foldername but not in the subfolder.

    #     # get the list of images in the foldername
    #     images_in_foldername = [x for x in foldername if buoy_id in x]
    #     # get the list of images in the subfolder
    #     images_in_subfolder = [x for x in os.listdir(f'images/panels/{buoy_id}/batches/{new_folder_name}') if buoy_id in x]
    #     # get the list of images that are in the foldername but not in the subfolder
    #     #!images_to_delete = [x for x in images_in_foldername if x not in images_in_subfolder]
    #     # delete the images that are in the foldername but not in the subfolder
    #     #for image in images_to_delete:
    #     #    os.remove(f'images/panels/{buoy_id}/{image}')

    #     # img1 = os.path.join('images','panels',str(buoy_id),file_name) # the image we want to move
    #     # img2 = os.path.join('images','batches',str(buoy_id),new_folder_name) # the folder we want to move the image to.
    #     # # i.e. move the image from images/panels/41002/2022_11_5_15_44_panel_1.png to images/panels/41002/2022_11_5_15/2022_11_5_15_44_panel_1.png
    #     # shutil.copy(img1, img2) # copy the image to the folder.
    pass


# 51101,51000,51003,51002,51004 - Hawaii
# 46006 - Amazing Sunset
# 46089 - Tillamook Oregon.

from difPy import dif

cam_urls = buoy_links()  # get the links to the cameras
all_buoy_urls = create_buoy_links(ids)
stitch_switch = False  # make false if you don't want to stitch the images.

# open the blacklist file

from ratelimit import limits, sleep_and_retry

# @limits(calls=1, period=4) # limit the number of calls to the function to 1 every 4 seconds.
@sleep_and_retry
def pull_data(cam_url, buoy_id, now):
    img = requests.get(cam_url)  # get the image
    if img.status_code == 200:
        return img
    else:
        print("status code", img.status_code, "for buoy", buoy_id)
    return img


### Testing the code
# detect red in an image
# load the image
image = cv2.imread("images/buoys/46072/2022_11_5_19_27.jpg")
image_path = "images/buoys/46072/2022_11_5_19_27.jpg"
# * Test 2.
# result = finding_red_version_two(image) # find the red in the image
# print(result)

# * Test 3. hsv and npwhere
output_img = finding_red_version_three(image_path)  # find the red in the image
print(output_img)
# print(output_hsv)


# Notes to self: remove functions for tests up to this point.
# * Test 4. Just red percent
# & Successful!
percent_red = detect_red_v4(image_path)
print(percent_red)


# test with the function to see if it detects red.

detect_red_v4(image_path)  # returns True if it detects red, False if it doesn't.


do_loop = True

if do_loop:
    pass
else:
    exit()  # exit the program if do_loop is False.


while True:
    try:
        # turn on at 4 am CST and turn off at 11 pm CST
        if (
            datetime.datetime.now().hour < 3 or datetime.datetime.now().hour > 24
        ):  # if it is before 3 am or after 12 am
            # wait to turn on until 4 am CST
            # keep the computer awake
            print("The computer is sleeping")
            time.sleep(240)  # sleep for 4 minutes
            continue

        # updated blacklist file
        blacklist = (
            open("data/blacklisted_buoy_ids.csv").read().splitlines()
        )  # get the list of buoy ids that are blacklisted.
        # parse blacklist to remove extra ' and " characters
        blacklist = [x.replace('"', "") for x in blacklist]
        blacklist = [x.replace("'", "") for x in blacklist]
        # create a blacklist list of strings from blacklist
        blacklist = [str(x) for x in blacklist][0].replace(" ", "").split(",")

        # # if the time is between 4 am and 11 am pacific time, then your wait_period is 100 seconds
        # if datetime.datetime.now().hour >= 4 and datetime.datetime.now().hour < 11:
        #     wait_period = 100
        # # if the time is between 11 am and 11 pm pacific time, then your wait_period is 600 seconds
        # if datetime.datetime.now().hour >= 11 and datetime.datetime.now().hour < 13:
        #     wait_period = 600 # 10 minutes
        # wait for 15 minutes
        wait_period = 600  # 10 minutes
        start_time = (
            datetime.datetime.now()
        )  # use this to calculate the next time to download images (every ten minutes)
        #!print('Starting the download loop at {}'.format(start_time))
        # print('I can still see things! Downloading images...')
        chunk_size = 30  # download 30 images at a time then pause for 10 seconds.
        chunk_size_current = (
            0  # the current number of images downloaded in the current chunk.
        )
        for cam_url in tqdm(cam_urls):
            # get the buoy id from the camera url
            buoy_id = re.search("station=(.*)", cam_url).group(1)
            if (
                buoy_id in blacklist
            ):  # if the buoy id is in the blacklist, then skip it.
                continue  # skip this buoy id
            # get the current time
            now = datetime.datetime.now()
            # create a directory for the buoy id if it doesn't already exist
            if not os.path.exists("images/buoys/{}".format(buoy_id)):
                os.makedirs("images/buoys/{}".format(buoy_id))
                ##logging.info("Created directory for buoy {}".format(buoy_id))

            # get the image
            ##logging.info("Checking buoy {}".format(buoy_id)) # log the buoy id
            if "images/buoys/{}/{}_{}_{}_{}_{}.jpg".format(
                buoy_id, now.year, now.month, now.day, now.hour, now.minute
            ) not in os.listdir(
                "images/buoys/{}".format(buoy_id)
            ):  # if the image has not already been downloaded
                time.sleep(
                    0.15
                )  # wait 0.25 seconds to avoid getting blocked by the server
                if (
                    chunk_size_current < chunk_size
                ):  # if we have not downloaded 30 images yet
                    chunk_size_current += 1  # add one to the chunk size
                else:
                    time.sleep(15)  # wait 15 seconds
                    chunk_size_current = 0  # reset the chunk size

                wait = True  # set the wait variable to true
                while wait:  # while we are waiting
                    try:  # try to get the image
                        img = pull_data(cam_url, buoy_id, now)  # download the image
                        wait = False
                    except Exception as e:
                        # print(e)
                        wait = True
                        time.sleep(1)
                        continue

                # check if the image is white

                # Print the name of the image we are downloading
                print(
                    "Downloading image: {}".format(
                        "images/buoys/{}/{}_{}_{}_{}_{}.jpg".format(
                            buoy_id, now.year, now.month, now.day, now.hour, now.minute
                        )
                    )
                )
                # save the image
                with open(
                    "images/buoys/{}/{}_{}_{}_{}_{}.jpg".format(
                        buoy_id, now.year, now.month, now.day, now.hour, now.minute
                    ),
                    "wb+",
                ) as f:
                    f.write(img.content)  # write the image to the file
                # check if the image is daytime or nighttime
                # ##logging.WARNING("Skipped night detection model for buoy {}".format(buoy_id))
                # if not is_it_daytime('images/buoys/{}/{}_{}_{}_{}_{}.jpg'.format(buoy_id, now.year, now.month, now.day, now.hour, now.minute)): # if it is nighttime
                # then we will delete the image
                # *print(f'Deleting image for buoy {buoy_id} because it is nighttime.')
                # *os.remove('images/buoys/{}/{}_{}_{}_{}_{}.jpg'.format(buoy_id, now.year, now.month, now.day, now.hour, now.minute))
                #    pass

                # ^ check image to see if it is just a white screen or not. If it is then we want to add this buoy id to the blacklist so that we don't download images from it anymore.

            else:
                print(
                    "Image already exists: {}".format(
                        "images/buoys/{}/{}_{}_{}_{}_{}.jpg".format(
                            buoy_id, now.year, now.month, now.day, now.hour, now.minute
                        )
                    )
                )
                pass  # if the image already exists, don't download it again

        ##logging.INFO("Beginning to panel images (line 24)") #! at {}".format(datetime.datetime.now()))
        # Save the panels to the images/panels directory
        list_of_buoys = os.listdir(
            "images/buoys"
        )  # get the list of buoy ids by their directory names

        # sample a random 20 extras from the
        print("Creating panels...")
        for buoy_id in tqdm(list_of_buoys):
            # get the list of images for the buoy
            # print(f'Paneling images for buoy {buoy_id}')
            if (
                buoy_id != ".DS_Store" and "." not in buoy_id
            ):  # if the buoy id is not a hidden file
                images = os.listdir("images/buoys/{}".format(buoy_id))
                # if the image has not already been used to create panels, create the panels and save them to the images/panels directory
                ##logging.info("Saving panels for buoy {}".format(buoy_id))
                for image in images:
                    # print(f'    Paneling image {image}')
                    # if the image is not None
                    if image == ".DS_Store" or image != "None":
                        continue
                    # If the panels directory for the buoy doesn't exist, create it.
                    if not os.path.exists("images/panels/{}".format(buoy_id)):
                        os.makedirs("images/panels/{}".format(buoy_id))
                    if (
                        "images/buoys/{}/{}".format(buoy_id, image)
                        in os.listdir("images/panels/{}".format(buoy_id))
                        and image == ".DS_Store"
                        and buoy_id != ".DS_Store"
                    ):
                        print(
                            "This image has already been used to create panels. Or it is a hidden file."
                        )
                    else:
                        # get the panels
                        (
                            panel_1,
                            panel_2,
                            panel_3,
                            panel_4,
                            panel_5,
                            panel_6,
                        ) = divide_into_panels(
                            buoy_id, "images/buoys/{}/{}".format(buoy_id, image)
                        )

                    # print('Processing image: {}'.format(image))

                    ##logging.info("Saved panels for buoy {}".format(buoy_id))
                    # print('Saving panels...')
                    # save the panels to the images/panels directory
                    # now, stitch these images together (correcting for the misalignment of the cameras) and save the result to the images/panoramas directory
                    # print('Stitching panels...')
                    # stitch the panels together using stitched_panoramas

                    # try:
                    #     stitched = stitched_panoramas(panel_1, panel_2, panel_3, panel_4, panel_5, panel_6)
                    #     # save the stitched panorama to the images/panoramas directory
                    #     # print('Saving stitched panorama...')
                    #     if not os.path.exists('images/panoramas/{}'.format(buoy_id)):
                    #         os.makedirs('images/panoramas/{}'.format(buoy_id))
                    #     # print('images/panoramas/{}/{}_{}_{}_{}_{}.jpg'.format(buoy_id, now.year, now.month, now.day, now.hour, now.minute))
                    #     cv2.imwrite('images/panoramas/{}/{}_{}_{}_{}_{}.jpg'.format(buoy_id, now.year, now.month, now.day, now.hour, now.minute), stitched)
                    #     # move the image to the images/panels directory
                    #     os.rename('images/buoys/{}/{}'.format(buoy_id, image), 'images/panels/{}/{}'.format(buoy_id, image))
                    #     ##logging.info("Moved image to panels directory for buoy {}".format(buoy_id))
                    # except Exception as e:
                    #     print(e)
                    #     pass

        # Stage 4: save buoy_update_rates_dict to a csv file
        buoy_update_rates_dict_df = pd.DataFrame.from_dict(
            buoy_update_rates_dict, orient="index"
        )
        buoy_update_rates_dict_df.to_csv("data/buoy_update_rates_dict.csv")

        # Stage 5: Using DifPy, find any images that are similar 'normal' to white_blank.jpg and delete them.
        # parse the buoy folders and their images

        try:
            buoy_folders = os.listdir("images/buoys")
            for buoy_folder in buoy_folders:
                if buoy_folder != ".DS_Store":
                    images = os.listdir("images/buoys/{}".format(buoy_folder))
                    for image in images:
                        if image != ".DS_Store" and image != "None":
                            # get the image path
                            image_path = "images/buoys/{}/{}".format(buoy_folder, image)
                            # get the image
                            image = cv2.imread(image_path)
                            white_image = cv2.imread("images/white_blank.jpg")
                            # we need these images to be the same size, so we will resize the white image to the size of the image
                            white_image = cv2.resize(
                                white_image, (image.shape[1], image.shape[0])
                            )
                            # are they ndarrays?
                            # print(type(image))
                            # print(type(white_image))

                            # get the difference between the image and the white_blank.jpg image
                            # calculate the difference between pixel values of the image and a pure white image using numpy
                            diff = np.sum(
                                np.abs(image - white_image)
                            )  # get the sum of the absolute difference between the two images
                            # if the difference is less than 1000, then we will delete the image
                            if diff < 1000:
                                print("Deleting image: {}".format(image_path))
                                # move the images instead of literally deleting them to a folder called 'deleted_images' in the images directory
                                if not os.path.exists("images/deleted_images"):
                                    os.makedirs("images/deleted_images")
                                os.rename(
                                    image_path,
                                    "images/deleted_images/{}_{}".format(
                                        image_path.split("/")[-1].split(".")[0],
                                        buoy_folder,
                                    ),
                                )
                                os.remove(image_path)
                            # get the difference score from the difference image
                            # difference_score = dif.get_difference_score()
                            # if the difference score is less than 0.1, then delete the image

                            # if the difference is less than 0.1, then delete the image
                            # if difference_score < 0.1:
                            #     os.remove(image_path)
                            #     print('Deleted image {} because it was too similar to white_blank.jpg'.format(image_path))
        except Exception as e:
            print("Error with White Image Detection: {}".format(e))
            pass

        if exper_1:
            # run DifPy on the images in the images/panels directory
            try:
                buoy_folders = os.listdir("images/panels")
                for buoy_folder in buoy_folders:
                    if buoy_folder != ".DS_Store":
                        images = os.listdir("images/panels/{}".format(buoy_folder))
                        for image in images:
                            if image != ".DS_Store":
                                # get the image path
                                image_path = "images/panels/{}/{}".format(
                                    buoy_folder, image
                                )
                                # get the image
                                image = cv2.imread(image_path)
                                white_image = cv2.imread("images/white_blank.jpg")
                                # we need these images to be the same size, so we will resize the white image to the size of the image
                                white_image = cv2.resize(
                                    white_image, (image.shape[1], image.shape[0])
                                )
                                # are they ndarrays?
                                # print(type(image))
                                # print(type(white_image))

                                # get the difference between the image and the white_blank.jpg image
                                # calculate the difference between pixel values of the image and a pure white image using numpy
                                diff = np.sum(
                                    np.abs(image - white_image)
                                )  # get the sum of the absolute difference between the two images
                                # if the difference is less than 1000, then we will delete the image
                                if diff < 1000:
                                    print("Deleting image: {}".format(image_path))
                                    # move the images instead of literally deleting them to a folder called 'deleted_images' in the images directory
                                    if not os.path.exists("images/deleted_images"):
                                        os.makedirs("images/deleted_images")
                                    os.rename(
                                        image_path,
                                        "images/deleted_images/{}_{}".format(
                                            image_path.split("/")[-1].split(".")[0],
                                            buoy_folder,
                                        ),
                                    )
                                    # os.remove(image_path)
                                # dif.
                                # # get the difference score from the difference image
                                # difference_score = dif.get_difference_score()
                                # # if the difference score is less than 0.1, then delete the image

                                # # if the difference is less than 0.1, then delete the image
                                # if difference_score < 0.1:
                                #     os.remove(image_path)
                                #     print('Deleted image {} because it was too similar to white_blank.jpg'.format(image_path))
            except Exception as e:
                print("Error with White Image Detection: {}".format(e))
                pass

        # Remove duplicate images (preferably before paneling but for now after)
        if duplicate_removal_flag == True:

            for folder in os.listdir("images/buoys"):
                if folder == ".DS_Store":
                    continue
                # get the list of images in the folder
                # sort the images by date
                # make folder_path variable from relative path
                folder_path = "images/buoys/{}".format(folder)
                search = dif(
                    folder_path,
                    similarity="high",
                    show_output=False,
                    show_progress=True,
                )  # returns a list of lists of similar images
                # for each list of similar images, move all but the first image to the deleted_images folder
                file_results_dict = search.result  # get the list of file names
                # {20220824212437767808 : {"filename" : "image1.jpg",
                #                         "location" : "C:/Path/to/Image/image1.jpg"},
                #                         "duplicates" : ["C:/Path/to/Image/duplicate_image1.jpg",
                #                                         "C:/Path/to/Image/duplicate_image2.jpg"]},
                # This is the format of the dictionary returned by the dif.search() method
                # I want to the filename, location, and duplicates
                # I want to move the duplicates to the deleted_images folder

                # make the deleted_images folder if it doesn't exist
                if not os.path.exists("images/deleted_images"):
                    os.makedirs("images/deleted_images")

                # counter should be how many files are in the deleted folder before we start
                counter = len(os.listdir("images/deleted_images"))
                # move the duplicates to the deleted_images folder
                for (
                    key
                ) in file_results_dict:  # iterate through the keys in the dictionary
                    # get the duplicates
                    value = file_results_dict[key]
                    duplicates = value["duplicates"]
                    for duplicate in duplicates:
                        try:
                            # move the duplicate to the deleted_images folder
                            # os.rename(duplicate, 'images/deleted_images/{}_{}'.format(counter,duplicate.split('/')[-1]))
                            # remove the duplicate
                            # full dupe path
                            # full_dupe_path = 'images/buoys/{}/{}'.format(folder, duplicate.split('/')[-1])

                            # first add "duplicate_" to the beginning of the file name
                            new_name = duplicate.split("/")[-1]  # get the file name
                            new_name = "duplicate_{}".format(
                                new_name
                            )  # add duplicate_ to the beginning of the file name
                            # then rename it in the same directory as the original
                            os.rename(
                                duplicate, "images/buoys/{}/{}".format(folder, new_name)
                            )
                            # then move the file to the deleted_images folder
                            print("Renamed {} to {}".format(duplicate, new_name))
                            # os.rename(duplicate, str(duplicate).replace('images/buoys', 'images/deleted_images'))
                            counter += 1
                        except Exception as e:
                            print("Error moving duplicate image: {}".format(e))
                            pass

        ignoring_panel_optimimal = True  # note: this is a temporary fix to the problem of the panel images not being generated
        # final step: make sure that all the previous buoy images have been panelled and saved to the images/panels directory
        for folder in tqdm(os.listdir("images/buoys")):
            # print('Checking if all images have been panelled for buoy {}'.format(folder))
            try:
                if folder == ".DS_Store":
                    continue
                images = os.listdir("images/buoys/{}".format(folder))
                # if the folder is not in the images/panels directory, then we need to panel the images
                # if not os.path.exists('images/panels/{}'.format(folder)):
                if ignoring_panel_optimimal:
                    # print('The images in the images/buoys/{} directory have not been panelled yet.'.format(folder))
                    # panelling the images
                    try:
                        os.mkdir("images/panels/{}".format(folder))
                        print(
                            "made directory for buoy {}".format(folder)
                            + " in images/panels"
                        )
                    except:
                        pass
                    batch_id = 1
                    for image in images:
                        # make a folder for the batch that has the same name as the image without the extension
                        try:
                            i_name = image[:-4]
                            directory_save_path = f"images/panels/{folder}/{i_name}"  # make the directory path
                            os.mkdir(directory_save_path)
                        except FileExistsError:
                            pass
                        # get the panels
                        # if the folder is not empty skip it
                        if len(os.listdir(directory_save_path)) > 0:
                            continue
                        try:
                            if image == ".DS_Store":
                                continue
                            # get the panels and save them to directory_save_path
                            (
                                panel_1,
                                panel_2,
                                panel_3,
                                panel_4,
                                panel_5,
                                panel_6,
                            ) = divide_into_panels(
                                folder, "images/buoys/{}/{}".format(folder, image)
                            )
                            # save the panels to the directory_save_path
                            # panel_1.save(f'{directory_save_path}/panel_1.jpg')
                            # panel_2.save(f'{directory_save_path}/panel_2.jpg')
                            # panel_3.save(f'{directory_save_path}/panel_3.jpg')
                            # panel_4.save(f'{directory_save_path}/panel_4.jpg')
                            # panel_5.save(f'{directory_save_path}/panel_5.jpg')
                            # panel_6.save(f'{directory_save_path}/panel_6.jpg')
                            ##logging.info('Saved panels for image: {}'.format(image))

                        except:
                            ##logging.ERROR('Error while saving panels for image: {}'.format(image))
                            # print('Could not create panels for image: {}'.format(image))
                            continue

                        # note: trying to add in the vincent code here
                        # stitch the images together
                        if stitch_switch:
                            files_to_stitch = [
                                f"{directory_save_path}/panel_1.jpg",
                                f"{directory_save_path}/panel_2.jpg",
                                f"{directory_save_path}/panel_3.jpg",
                                f"{directory_save_path}/panel_4.jpg",
                                f"{directory_save_path}/panel_5.jpg",
                                f"{directory_save_path}/panel_6.jpg",
                            ]  # list of files to stitch

                            # Stitch the images together with OpenCV and save the stitched image to the panoramas directory
                            print("Stitching images...")
                            try:
                                ocean_stitching(
                                    files_to_stitch,
                                    f"images/panoramas/{folder}/{i_name}.jpg",
                                )  # stitch the images together and save the stitched image to the panoramas directory
                            except Exception as f:
                                print(f)
                                print(
                                    "Could not stitch images for image: {}".format(
                                        image
                                    )
                                )
                            # > Overload resolution failed:
                            # >  - Can't parse 'images'. Sequence item with index 0 has a wrong type
                            # >  - Can't parse 'images'. Sequence item with index 0 has a wrong type
                            # >  - Stitcher.stitch() missing required argument 'masks' (pos 2)
                            # >  - Stitcher.stitch() missing required argument 'masks' (pos 2)
                            # fix: https://stackoverflow.com/questions/6380057/python-cv2-error-215-overload-resolution-failed

                        # try:
                        #     print('Stitching images for image set {}'.format(files_to_stitch))
                        #     panorama = vincent.make_panorama(images)
                        #     #remove the .jpg from the image name
                        #     image_name = image[:-4] # remove the .jpg from the image name

                        #     cv2.imwrite(f'images/panoramas/{folder}_{image_name}_panorama.png', panorama)
                        # except Exception as e:
                        #     print(e)
                        #     print('Could not stitch images together for buoy: {}'.format(folder))
                        #     continue

                        batch_id += 1
            except Exception as e:
                print(e)
                # print('Could not create panels for buoy: {}'.format(folder))
                # print('The images in the images/buoys/{} directory have not been panelled yet.'.format(folder))
                # print('line 139') # line 139
                pass
        # for each folder in the images/panels folder, stitch the images together and save them to the images/panoramas folder with the same name as the folder + panorama.png

        # //: the for loop below does not account for the fact that there are multiple captures with 6 frames per capture. This means that the images will be stitched together incorrectly. This is a problem that needs to be fixed. Find a way to select only the sets of 6 images that go together to stitch together.

        print("stage 5 complete")
        # Stage 6: Create the buoy dataframes
        # if it has been ten minutes since the last time the data was fetched, fetch the data again
        if time.time() - last_time_fetched > 600 or first_run:
            latest_data = (
                get_latest_data()
            )  # get the latest data from the RSS feed (updates every 10 minutes)
            # save the master dataframe to a csv file
            run_date = time.strftime("%Y%m%d_%H%M%S")
            latest_data.to_csv(f"data/rss/rss_buoy_data_{run_date}.csv", index=False)
            print("Done with this run")
            time_last_fetched_rss = time.time()  # get the time of the last fetch
        print("stage 6 complete")

        # * ======= show the last buoy image captured in this run
        try:
            display_last_image(
                list_of_buoys
            )  # display the last image captured in this run
        except:
            pass
        # * ====== End Show Buoy Image Snippet

        print("Running White Elimiination and populating the Tapestry")
        deal_with_white_images_and_populate_tapestry()  # run the white elimination and populate the tapestry

        # how much time has passed since the start of the loop?
        time_elapsed = datetime.datetime.now() - start_time
        # wait until the time elapsed is 15 minutes from the start of the loop
        print("Waiting for the remainder of the minutes...")

        if verbose_wait:
            # wait_period = 100 # was 900 (15 minutes)
            for i in tqdm(range(wait_period - time_elapsed.seconds)):
                time.sleep(1)
            iteration_counter += 1
        else:
            print(
                "Waiting for the remaining {} seconds".format(
                    wait_period - time_elapsed.seconds
                )
            )
            time.sleep(wait_period - time_elapsed.seconds)
            iteration_counter += 1

    except Exception as e:
        print(e)
        print("Error occurred.")
        # how much time has passed since the start of the loop?
        time_elapsed = datetime.datetime.now() - start_time
        # * wait till the ten minute mark is reached.
        for i in tqdm(range(wait_period - time_elapsed.seconds)):
            time.sleep(1)
        iteration_counter += 1

    # # now we have all the images in the subfolders
    # # we want to return a list of lists of images that are in the subfolders

    # list_of_lists = []
    # for folder in os.listdir(f'images/batches/{buoy_id}'):
    #     try:
    #         # Get the list of images in the folder, and append it to the list of lists (these are the relative paths to the images)
    #         list_of_lists.append(os.listdir(f'images/batches/{buoy_id}/{folder}')) # this will be a list of 6 images
    #     except NotADirectoryError:
    #         # there is a file in the folder
    #         continue
    #     except FileNotFoundError:
    #         # there is a missing folder
    #         print(f'FileNotFoundError: {folder}')
    #         continue
    #     except Exception as e:
    #         print(e)
    #         continue
    # return list_of_lists # return a list of sets of images that have the same buoy id, year, month, day, hour, and panel numbers 1,2,3,4,5,6


# Next Steps

"""

The next steps are to:
Use the Keras model saved in the models folder to predict the content of each panel image as it is created, and save the predictions to a csv file.
These predictions will determine which buoys the model will follow closest and which buoys it will eventually ignore. This will be done by finding which of the buoy cameras record the most interesting content (i.e. the most interesting content will be the content that the model predicts as being the most interesting).



            # daytime images always have higher values than 10 for all three channels
            # values less than 10 are usually night
            # skip the image if it is night
            #if orange_value < 10 and red_value < 10: # if the image is night
            #    continue
            # if the values are all higher than 250 then it is a white image and we want to remove it
            #if orange_value > 250 and red_value > 250:
            #    os.remove(file)
            #    print("Removed white image")
            #    continue
            # interesting photos have high red values
            #if red_value > 147 or (orange_value > 147 and #red_value > 100): # if the image is interesting
            #    add_list.append(file)
            # if the image was not taken in the last x hours, skip it

"""

It looks like your code is using OpenCV to stitch together images from buoys, but there are several issues with the code. Firstly, it is not clear how the images are being fetched from the buoys. It appears that the code is using a function called cam_backend(), but this function is not defined in the code provided. Additionally, the code is missing import statements for some of the modules it uses, such as time, glob, and os. These errors will need to be fixed before the code can be run.

Once the code is fixed and the images are being fetched properly, you can use OpenCV's stitching module to automatically stitch the images together. The stitching module can align the images and blend them seamlessly to create a panoramic photo.

To use the stitching module, you will need to first load the images using the cv2.imread() function and then pass them to the cv2.Stitcher_create() function to create a stitcher object. Then, you can use the stitcher.stitch() method to stitch the images together. This method returns a tuple containing the stitched image and a status code indicating if the stitching was successful or not.

Once the stitched image is created, you can use OpenCV's computer vision algorithms to detect and classify the images as sunsets. This can be done using a pre-trained machine learning model that has been trained to recognize sunsets in images. To use the model, you will need to load the image using the cv2.imread() function and then pass it to the model for prediction. The model will return a label indicating whether the image is a sunset or not.

Finally, once the sunset images have been identified, you can display them on the main project page for others to see. This can be done using a web framework such as Flask or Django, or by simply saving the images to a folder and displaying them using HTML.

Overall, the key steps in this project involve using the NOAA API to pull images from the buoys, using OpenCV to stitch the images together and classify them, and displaying the resulting panoramic photos and sunset images on the main project page.




Graham Waters
Okay, so I want to start this project from square one.

If we restarted with a blank .py file what would you import first?

If you are starting with a blank .py file and want to use OpenCV to stitch together images from buoys and classify them as sunsets, you will need to import the following modules:

cv2: This is the main OpenCV module, which provides functions for loading, processing, and saving images.
argparse: This module is used to parse command-line arguments, which can be used to specify the input and output filenames for the images.
imutils: This module provides convenience functions for resizing, rotating, and flipping images.
numpy: This module provides support for multi-dimensional arrays, which are used to represent images in OpenCV.
To import these modules, you can use the following code at the top of your .py file:

Copy code
import cv2
import argparse
import imutils
import numpy as np
Once the necessary modules have been imported, you can use the cv2.imread() function to load the images from the buoys, and the cv2.Stitcher_create() function to create a stitcher object that can be used to stitch the images together. The stitched image can then be passed to a pre-trained machine learning model to classify it as a sunset or not.

Overall, the key steps in this project involve using the NOAA API to pull images from the buoys, using OpenCV to stitch the images together and classify them, and displaying the resulting panoramic photos and sunset images on the main project page.




Graham Waters
okay I imported the libraries like you suggested. What next?

After importing the necessary modules, the next step is to use the NOAA API to pull images from the buoys. This can be done using the requests module, which provides functions for making HTTP requests. To use the requests module, you will need to first install it using the following command:

Copy code
pip install requests
Once the requests module is installed, you can use the requests.get() function to make a request to the NOAA API and retrieve the images from the buoys. The requests.get() function takes the URL of the API endpoint as an argument and returns a Response object containing the response data.

To retrieve the images from the buoys, you can use the following code:

Copy code
import requests

# make a request to the NOAA API endpoint
response = requests.get("https://api.noaa.gov/endpoint")

# check the response status code
if response.status_code == 200:
    # if the request was successful, retrieve the images from the response
    images = response.json()["images"]
else:
    # if the request was unsuccessful, print an error message
    print("Error retrieving images from the NOAA API")
Once the images are retrieved from the NOAA API, you can use the cv2.imread() function to load the images from their filenames. This function takes the filename of the image as an argument and returns the image data as a numpy array.

To load the images, you can use the following code:

Copy code
# create an empty list to store the image data
image_data = []

# for each of the images
for image in images:
    # load the image data
    img = cv2.imread(image["filename"])
    # append the image data to the list
    image_data.append(img)
Once the image data is loaded, you can use the cv2.Stitcher_create() function to create a stitcher object that can be used to stitch the images together. The stitcher object contains algorithms for aligning and blending the images to create a panoramic photo.

To create the stitcher object, you can use



·
Free Research Preview: ChatGPT is optimized for dialogue. Our goal is to make AI systems more natural to interact with, and your feedback will help us improve our systems and make them safer.
Reset Thread
Light Mode
OpenAI Discord
Learn More
Log out


Graham Waters
Project PySeas I want to make a beautiful, evolving art piece out of the images generated every hour by the buoys in the ocean, which the National Oceanographic Atmospheric Association or NOAA maintains. They have an API that allows you to call that API and pull images from buoys worldwide. I want to be able to take those images every hour from the buoys and stitch them together using Open CV in a way that makes it I would like to be able to stitch them together to create a panoramic photo, but not by just concatenating the images because that doesn't always look good. Also, I would like to match the images' horizon lines because the buoy images are uneven due to the waves. It is usually rocking back and forth. Another goal of this project is to find the most beautiful sunset panoramic photos and display those on the main project page. So that involves a little computer vision and classifying images as sunsets. I have some code already and need to refactor it.

It sounds like you have a great idea for a project! Using OpenCV for image stitching and horizon line matching is a good approach. One way to accomplish this would be to use feature matching to find corresponding points in the images from the buoys, and then use those points to warp and align the images. This would allow you to create a panoramic photo that takes into account the waves and uneven horizon lines.

To classify the images as sunsets, you could use a machine learning model trained on a dataset of sunset images. This model could then be used to predict whether a given image is a sunset or not. Once you have identified the sunset images, you can display them on the main project page.

As for refactoring your existing code, it would be helpful to see the code and understand your specific goals and challenges in order to provide more specific suggestions. In general, refactoring code involves making changes to the structure and organization of the code to improve its readability, maintainability, and performance. This often involves identifying and addressing issues such as duplicated code, complex or poorly-named functions, and suboptimal algorithms.




Graham Waters
my files are so disorganized. Here is my directory.
# old

* [MANIFEST.in](./old/MANIFEST.in)
* [Makefile](./old/Makefile)
* [PyBuoy.py](./old/PyBuoy.py)
* [cam_backend.py](./old/cam_backend.py)
* [cam_finder_raw.py](./old/cam_finder_raw.py)
* [camfinder.py](./old/camfinder.py)
* [carmensandiego.py](./old/carmensandiego.py)
* [finding_sunrise.py](./old/finding_sunrise.py)
* [looper.py](./old/looper.py)
* [one.py](./old/one.py)
* [readme.md](./old/readme.md)
* [readme_helperfunctions.py](./old/readme_helperfunctions.py)
* [reformat.py](./old/reformat.py)
* [requirements.txt](./old/requirements.txt)
* [setup.py](./old/setup.py)
* [tempfile.txt](./old/tempfile.txt)
* [test1.json](./old/test1.json)
* [using_keras_modeling.py](./old/using_keras_modeling.py)

It looks like your directory contains a mix of files with different purposes and formats, such as Python scripts, a Makefile, and a requirements file. It can be helpful to organize these files into separate folders based on their function or type. For example, you could create a src folder for your Python scripts, a data folder for any data files you are using, and a model folder for any trained machine learning models. This can make it easier to find and manage your files, and also improve the readability and maintainability of your project.

Another thing you may want to consider is using a version control system such as Git to manage your code and track changes. This can help you keep track of different versions of your code, collaborate with others on the project, and roll back to previous versions if needed. You can learn more about Git and how to use it by following online tutorials or reading the documentation.




Graham Waters
Here's my code:

from cam_backend import *

print("Starting the download loop")
last_time_fetched = time.time()  # get the current time
first_run = True  # set a flag to indicate that this is the first run of the loop (for the first run, we will download rss feeds for all the buoys)
duplicate_removal_flag = (
    True  # set this flag to true if we want to remove duplicated images with difPy
)
# note: bugs are present in difPy, so this flag is set to false
exper_1 = False  # flag for dupe detect in panels
verbose_wait = False  # flag that makes waiting show a progress bar.

import glob

import cv2
import imutils
# import the necessary packages
import numpy as np

rotating = True  # flag to indicate if the tapestry is rotating
panel_mode = False  # flag to indicate if we want to use panels for color detection


def is_recent(file, minutes):
    # get the time the image was taken
    image_time = os.path.getmtime(file)
    # get the current time
    current_time = time.time()
    # get the difference between the two times
    time_difference = current_time - image_time
    # if the time difference is less than minutes, return true
    if time_difference < minutes * 60:
        return True
    else:
        return False


def crop_the_bottom_off(images):
    # for each of the images crop the bottom off (20 pixels)
    for image in images:
        try:
            # get the image size
            img_width, img_height = get_image_size(image)
            # crop the bottom off
            cropped_image = image.crop((0, 0, img_width, img_height - 20))
            # save the image
            cropped_image.save(image)
        except Exception as e:
            print("Error cropping the bottom off of the image: " + str(e))


def check_colors_of_six_panels(image):
    # there are six panels in the image (side by side) and we want to check the colors of each panel
    # get the image size
    # the input image should be an image object

    img_width, img_height = get_image_size(image)
    # get the width of each panel
    panel_width = img_width / 6
    # get the height of each panel (remove botttom 20 pixels)
    panel_height = img_height - 20
    # get the colors of each panel
    panel_1 = image.getpixel((panel_width / 2, panel_height / 2))
    panel_2 = image.getpixel((panel_width * 1.5, panel_height / 2))
    panel_3 = image.getpixel((panel_width * 2.5, panel_height / 2))
    panel_4 = image.getpixel((panel_width * 3.5, panel_height / 2))
    panel_5 = image.getpixel((panel_width * 4.5, panel_height / 2))
    panel_6 = image.getpixel((panel_width * 5.5, panel_height / 2))
    # return the number of panels that meet the color criteria:
    # criteria:
    # red = less than 250 and greater than 170
    # green = less than 250 and greater than 170
    # blue = less than 250 and greater than 170
    # orange = less than 250 and greater than 170

    # check panel 1
    # if panel_1[0] < 250 and panel_1[0] > 170 and panel_1[1] < 250 and panel_1[1] > 170 and panel_1[2] < 250 and panel_1[2] > 170:
    #     panel_1_color = 'white'
    # elif panel_1[0] < 250 and panel_1[0] > 170 and panel_1[1] < 250 and panel_1[1] > 170 and panel_1[2] < 170:
    #     panel_1_color = 'yellow'
    # elif panel_1[0] < 250 and panel_1[0] > 170 and panel_1[1] < 170 and panel_1[2] < 250 and panel_1[2] > 170:
    #     panel_1_color = 'green'
    # elif panel_1[0] < 170 and panel_1[1] < 250 and panel_1[1] > 170 and panel_1[2] < 250 and panel_1[2] > 170:
    #     panel_1_color = 'blue'
    # elif panel_1[0] < 250 and panel_1[0] > 170 and panel_1[1] < 170 and panel_1[2] < 170:
    #     panel_1_color = 'red'
    # else:
    #     panel_1_color = 'other'
    # this is an interesting code but let's do something simpler, I want to isolate images that have sunsets in them, so let's just check the red and blue values
    if (
        panel_1[0] > 200 and panel_1[2] > 200
    ):  # if the red and blue values are greater than 200, then it's a sunset?
        panel_1_result = True  # set the result to true
    else:
        panel_1_result = False
    # check panel 2
    if panel_2[0] > 200 and panel_2[2] > 200:
        panel_2_result = True
    else:
        panel_2_result = False
    # check panel 3
    if panel_3[0] > 200 and panel_3[2] > 200:
        panel_3_result = True
    # check panel 4
    if panel_4[0] > 200 and panel_4[2] > 200:
        panel_4_result = True
    # check panel 5
    if panel_5[0] > 200 and panel_5[2] > 200:
        panel_5_result = True
    # check panel 6
    if panel_6[0] > 200 and panel_6[2] > 200:
        panel_6_result = True
    # return the results
    panels_collection = [
        panel_1,
        panel_2,
        panel_3,
        panel_4,
        panel_5,
        panel_6,
    ]  # put the panels into a list
    return (
        panel_1,
        panel_2,
        panel_3,
        panel_4,
        panel_5,
        panel_6,
        panel_1_result,
        panel_2_result,
        panel_3_result,
        panel_4_result,
        panel_5_result,
        panel_6_result,
    )


def get_panel_segments(image):
    """
    get_panel_segments takes an image and returns the segments of the image that are the panels

    :param image: the image to be segmented

    :param image: the image to be segmented
    :type image: image object (from PIL) or numpy array (from OpenCV)
    :return: the segments of the image that are the panels
    :rtype: list of image objects
    """
    # get the image size
    img_width, img_height = image.shape[0], image.shape[1]
    # get the width of each panel
    panel_width = img_width / 6
    # get the height of each panel (remove botttom 20 pixels)
    panel_height = img_height - 20
    # cast them to integers
    panel_width = int(panel_width)
    panel_height = int(panel_height)

    # get the segments of the image that are the panels
    panel_1 = image[
        0:panel_height, 0:panel_width
    ]  # to overcome the error: slice indices must be integers or None or have an __index__ method
    # we have to convert the panel_1 to an image object
    panel_2 = image[0:panel_height, panel_width : panel_width * 2]
    panel_3 = image[0:panel_height, panel_width * 2 : panel_width * 3]
    panel_4 = image[0:panel_height, panel_width * 3 : panel_width * 4]
    panel_5 = image[0:panel_height, panel_width * 4 : panel_width * 5]
    panel_6 = image[0:panel_height, panel_width * 5 : panel_width * 6]
    # return the segments
    panels = [panel_1, panel_2, panel_3, panel_4, panel_5, panel_6]
    return panels


def get_average_color(image):
    """
    get_average_color takes an image and returns the average color of the image

    :param image: the image to be segmented
    :type image: image object (from PIL) or numpy array (from OpenCV)
    :return: the average color of the image
    :rtype: tuple of integers
    """
    # get the image size
    img_width, img_height = image.shape[0], image.shape[1]
    # get the average color of the image
    # to do this we have to convert the image from a numpy array to a PIL image
    image = Image.fromarray(image)
    average_color = image.getpixel((img_width // 2, img_height // 2))
    # return the average color
    return average_color


def generate_one_full_day(buoy_id, day_number, month_number, year_number):
    # concatenate one image for each hour of the day and save it as a new image called 'full_day.png'
    # get the images from the buoy_id folder and sort them
    images = os.listdir(buoy_id)
    images.sort()
    # the names of the images are in the format: '2019-01-01_00.png'
    # images/buoys/41004/2022_11_5_15_44.jpg
    # where the day is the 3rd element in the list
    # and the hour is the 4th element in the list
    # so we can use a list comprehension to get the images for each hour of the day if we sort the images first.
    # go through the folder and add the filenames to the list if the day is the same as the day_number
    # and the month is the same as the month_number
    # and the year is the same as the year_number

    # get the images for the day
    images_for_day = [
        image
        for image in images
        if image.split("_")[2] == str(day_number)
        and image.split("_")[1] == str(month_number)
        and image.split("_")[0] == str(year_number)
    ]
    # get the images for the day and sort them
    images_for_day.sort()
    # get one image for each hour of the day
    images_for_day = [
        image
        for image in images_for_day
        if image.split("_")[3] == "00"
        or image.split("_")[3] == "01"
        or image.split("_")[3] == "02"
        or image.split("_")[3] == "03"
        or image.split("_")[3] == "04"
        or image.split("_")[3] == "05"
        or image.split("_")[3] == "06"
        or image.split("_")[3] == "07"
        or image.split("_")[3] == "08"
        or image.split("_")[3] == "09"
        or image.split("_")[3] == "10"
        or image.split("_")[3] == "11"
        or image.split("_")[3] == "12"
        or image.split("_")[3] == "13"
        or image.split("_")[3] == "14"
        or image.split("_")[3] == "15"
        or image.split("_")[3] == "16"
        or image.split("_")[3] == "17"
        or image.split("_")[3] == "18"
        or image.split("_")[3] == "19"
        or image.split("_")[3] == "20"
        or image.split("_")[3] == "21"
        or image.split("_")[3] == "22"
        or image.split("_")[3] == "23"
    ]
    # get the images for the day and sort them
    images_for_day.sort()
    # add the images one after the other by concatenating them vertically and save the result as a new image called 'full_day.png' in the buoy_id folder
    # get the first image
    first_image = images_for_day[0]
    # open the first image
    first_image = Image.open(buoy_id + "/" + first_image)
    # get the width and height of the first image
    width, height = first_image.size
    # create a new image with the same width as the first image and the height of the first image multiplied by the number of images
    new_image = Image.new("RGB", (width, height * len(images_for_day)))
    # paste the first image in the new image
    new_image.paste(first_image, (0, 0))
    # go through the rest of the images and add them to the bottom of the new image one after the other
    for i in range(1, len(images_for_day)):
        # get the image
        image = images_for_day[i]
        # open the image
        image = Image.open(buoy_id + "/" + image)
        # paste the image in the new image
        new_image.paste(image, (0, height * i))
    # save the new image
    new_image.save(buoy_id + "/full_day.png")


### Testing To find Red


def finding_red_version_two(image):
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image_height = image.shape[0]
    image_width = image.shape[1]
    # https://stackoverflow.com/questions/30331944/finding-red-color-in-image-using-python-opencv
    image_result = np.zeros((image_height, image_width, 3), np.uint8)
    for i in range(image_height):  # those are set elsewhere
        for j in range(image_width):  # those are set elsewhere
            if (
                img_hsv[i][j][1] >= 50
                and img_hsv[i][j][2] >= 50
                and (img_hsv[i][j][0] <= 10 or img_hsv[i][j][0] >= 170)
            ):
                image_result[i][j] = img_hsv[i][
                    j
                ]  # this is the red (above is saturation, value, and hue)
    return image_result


def finding_red_version_three(image_path):
    """
    finding_red_version_three takes an image and returns the red pixels in the image

    :param image_path: the path to the image to be segmented

    :param image_path: the path to the image to be segmented
    :type image_path: string
    :return: the red pixels in the image
    :rtype: image object
    """
    img = cv2.imread(image_path)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # lower mask (0-10)
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])
    mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

    # upper mask (170-180)
    lower_red = np.array([170, 50, 50])
    upper_red = np.array([180, 255, 255])
    mask1 = cv2.inRange(img_hsv, lower_red, upper_red)

    # join my masks
    mask = mask0 + mask1

    # set my output img to zero everywhere except my mask
    # output_img = img.copy()
    # output_img[np.where(mask==0)] = 0

    # or your HSV image, which I *believe* is what you want
    output_hsv = img_hsv.copy()
    output_hsv[np.where(mask == 0)] = 0

    # save the output_img to 'test.png' in images folder
    # cv2.imwrite('images/test.png', output_img)
    # save the output_hsv to 'test_hsv.png' in images folder
    cv2.imwrite("images/test_hsv.png", output_hsv)
    # why is the output_hsv image all black?
    # because the output_hsv image is in HSV format and not RGB format
    # so we have to convert it to RGB format
    # https://stackoverflow.com/questions/15007348/convert-hsv-to-rgb-using-python-and-opencv
    # converting the image from HSV to RGB
    output_hsv = cv2.cvtColor(output_hsv, cv2.COLOR_HSV2RGB)
    # save the output_hsv to 'test_hsv.png' in images folder
    cv2.imwrite("images/test_hsv.png", output_hsv)


def detect_red_v4(image):
    # Red color
    if type(image) == str:
        image = cv2.imread(image)  # read the image
    low_red = np.array([161, 155, 84])
    high_red = np.array([179, 255, 255])
    red_mask = cv2.inRange(image, low_red, high_red)
    percent_pixels_red = np.sum(red_mask) / (image.shape[0] * image.shape[1])
    return percent_pixels_red


def detect_red(img):
    """
    detect_red _summary_

    _extended_summary_

    :param image: _description_
    :type image:
    :return: _description_
    :rtype: _type_
    """
    try:
        # hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # creating a mask to catch the color red in the image
        # Here, you define your target color as
        # a tuple of three values: RGB
        # red = [130, 158, 0]
        red = [0, 0, 255]  # this is the color of the red in the image

        # You define an interval that covers the values
        # in the tuple and are below and above them by 20
        diff = 20

        # Be aware that opencv loads image in BGR format,
        # that's why the color values have been adjusted here:
        boundaries = [
            (
                [red[2], red[1] - diff, red[0] - diff],
                [red[2] + diff, red[1] + diff, red[0] + diff],
            )
        ]

        # Scale your BIG image into a small one:
        scalePercent = 0.3

        # Calculate the new dimensions
        width = int(img.shape[1] * scalePercent)
        height = int(img.shape[0] * scalePercent)
        newSize = (width, height)

        # Resize the image:
        img = cv2.resize(img, newSize, None, None, None, cv2.INTER_AREA)

        # check out the image resized:
        #!cv2.imshow("img resized", img)
        #!cv2.waitKey(0)

        # for each range in your boundary list:
        for (lower, upper) in boundaries:

            # You get the lower and upper part of the interval:
            lower = np.array(lower, dtype=np.uint8)
            upper = np.array(upper, dtype=np.uint8)

            # cv2.inRange is used to binarize (i.e., render in white/black) an image
            # All the pixels that fall inside your interval [lower, uipper] will be white
            # All the pixels that do not fall inside this interval will
            # be rendered in black, for all three channels:
            mask = cv2.inRange(img, lower, upper)

            # Check out the binary mask:
            #!cv2.imshow("binary mask", mask)
            # cv2.waitKey(0)

            # Now, you AND the mask and the input image
            # All the pixels that are white in the mask will
            # survive the AND operation, all the black pixels
            # will remain black
            output = cv2.bitwise_and(img, img, mask=mask)

            # Check out the ANDed mask:
            #!cv2.imshow("ANDed mask", output)
            # cv2.waitKey(0)

            # You can use the mask to count the number of white pixels.
            # Remember that the white pixels in the mask are those that
            # fall in your defined range, that is, every white pixel corresponds
            # to a red pixel. Divide by the image size and you got the
            # percentage of red pixels in the original image:
            ratio_red = cv2.countNonZero(mask) / (img.size / 3)

            # This is the color percent calculation, considering the resize I did earlier.
            colorPercent = (ratio_red * 100) / scalePercent

            # Print the color percent, use 2 figures past the decimal point
            print("red pixel percentage:", np.round(colorPercent, 2))

            # numpy's hstack is used to stack two images horizontally,
            # so you see the various images generated in one figure:
            #!cv2.imshow("images", np.hstack([img, output]))
            # save the image as 'test.png' in the images folder
            cv2.imwrite("images/test.png", np.hstack([img, output]))
            # cv2.waitKey(0)
    except Exception as e:
        print("Error in detect_red: ", e)


def deal_with_white_images_and_populate_tapestry():
    sunsets_found = 0  # keep track of how many sunsets we find
    files = glob.glob("images/buoys/*/*")
    # without glob
    # files = []
    # for file in os.listdir('images/buoys/'):
    #   files.append('images/buoys/' + file)
    # print(files)
    height, width, channels = cv2.imread(files[0]).shape
    # blank_image = np.zeros((height*10, width, channels), np.uint8)
    # get the ten images that have the most orange in them
    # make the blank image the same size as the images

    # shuffle the files so we don't always get the same ten images
    #!random.shuffle(files) #note: this could be a problem later

    add_list = []

    for file in tqdm(files):
        # read the image
        try:
            image = cv2.imread(file)
            if not is_recent(file, 300):  # 300 minutes
                continue

            if panel_mode:
                # get the image details for panels 1-6
                (
                    panel_1,
                    panel_2,
                    panel_3,
                    panel_4,
                    panel_5,
                    panel_6,
                ) = get_panel_segments(image)
                # explanation of results:
                panels_collection = [
                    panel_1,
                    panel_2,
                    panel_3,
                    panel_4,
                    panel_5,
                    panel_6,
                ]  # put the panels into a list

                # put True into panel_results once for each panel (1/6 of the width of the image) that has an average red value greater than 180 and an average blue value greater than 180

                # the image passed to get_panel_segments should be a cv2 image
                assert type(image) == np.ndarray

                panel_segments = get_panel_segments(image)

                positive_panel_count = (
                    0  # keep track of how many panels have a sunset in them
                )
                # get the average color of each panel
                for panel in panel_segments:
                    panel_average_color = get_average_color(panel)
                    # check if the panel is a sunset
                    if panel_average_color[0] > 200 and panel_average_color[2] > 200:
                        # increment the positive_panel_count by 1
                        positive_panel_count += 1

                # now check if the positive_panel_count is greater than 3 (i.e. more than half of the panels have a sunset in them)
                if positive_panel_count > 3:
                    add_list.append(file)
                    sunsets_found += 1
                else:
                    continue  # if the positive_panel_count is not greater than 3, then continue on to the next image

                # what is the average amount of red in the image?
                # what is the average amount of blue in the image?
                # what is the average amount of green in the image?
                # what is the average amount of orange in the image?

                # x = [[panel_1_result, panel_2_result, panel_3_result, panel_4_result, panel_5_result, panel_6_result],panels_collection] # return the results
                # panel_results - a list of true or false values for each panel (true if the panel is orange, false if not).
                # panels_collection - a list of the colors of each panel (in RGB format) (this is for debugging purposes)
                # if the image has at least 4 panels that are orange, then we want to add it to the tapestry

                # note: uncomment below if the check_colors_of_six_panels function is not working
                # # get the average orange value
                # orange_value = np.mean(image[:,:,2]) # the orange channel is the third channel
                # red_value = np.mean(image[:,:,0]) # get the average red value when over 147 then save the image

                # get the median orange value across the panels
                panels_collection = panel_segments  # put the panels into a list
                orange_value = np.median(
                    [
                        panels_collection[0][2],
                        panels_collection[1][2],
                        panels_collection[2][2],
                        panels_collection[3][2],
                        panels_collection[4][2],
                        panels_collection[5][2],
                    ]
                )
                # # get the median red value across the panels
                red_value = np.median(
                    [
                        panels_collection[0][0],
                        panels_collection[1][0],
                        panels_collection[2][0],
                        panels_collection[3][0],
                        panels_collection[4][0],
                        panels_collection[5][0],
                    ]
                )

            # if the average amount of orange is greater than 200, then add the image to the add_list

            # if the average amount of orange is greater than 200:
            # add_list.append(file)
            # sunsets_found += 1
            # else:
            # continue # if the average amount of orange is not greater than 200, then continue on to the next image

            # * just check the image to see if red is less than 20, green is less than 20, and blue is less than 20
            # * if so then skip the image

            red_score = np.mean(image[:, :, 0])
            green_score = np.mean(image[:, :, 1])
            blue_score = np.mean(image[:, :, 2])
            if red_score < 20 and green_score < 20 and blue_score < 20:
                # print('Night image detected')
                continue
            else:
                # print('Day image detected')
                red_val = detect_red_v4(image)
                if red_val > 2.5:
                    print(
                        " ---- > Sunset detected? ", red_val
                    )  # print the sunset detected message
                    # save the image to the sunset folder under the appropriate buoy
                    buoy_name = file.split("/")[2]
                    buoy_folder = "images/sunsets/" + buoy_name
                    if not os.path.exists(buoy_folder):
                        os.makedirs(buoy_folder)
                    cv2.imwrite(buoy_folder + "/" + file.split("/")[3], image)
                    red_flag = True
                elif red_val > 15:
                    print(
                        " ---- > super sunset detected? ", red_val
                    )  # print the sunset detected message
                    # save the image to the keepers folder under the appropriate buoy
                    buoy_name = file.split("/")[2]
                    buoy_folder = "images/keepers/" + buoy_name
                    if not os.path.exists(buoy_folder):
                        os.makedirs(buoy_folder)
                    cv2.imwrite(buoy_folder + "/" + file.split("/")[3], image)
                    red_flag = True
                else:
                    red_flag = False
                if red_flag:  # if the image has more than 10% red in it
                    add_list.append(file)
                    sunsets_found += 1

            # blue_value = np.mean(image[:,:,1]) # blue value
            # print(orange_value, red_value, blue_value)
            # show the image and annotate it with the orange, red, and blue values
            # plt.imshow(image)
            # plt.title("Orange: " + str(orange_value) + " Red: " + str(red_value) + " Blue: " + str(blue_value))
            # plt.show()

            # save the filename to a list if the image is to be added to the tapestry

        except Exception as e:
            print(e)
            continue

        blank_image = np.zeros((height * len(add_list), width, channels), np.uint8)
        try:
            cv2.imwrite("images/tapestry.png", blank_image)
        except Exception as e:
            print(e)
            print("Could not write blank image")
            print("line 322")
            continue

    for file in tqdm(add_list):
        # read the image
        try:
            image = cv2.imread(file)
            # get the average orange value
            # print(np.mean(image[:,:,2]))
            # orange_value = np.mean(image[:,:,2]) # the orange channel is the third channel

            # red_value = np.mean(image[:,:,0]) # get the average red value when over 147 then save the image

            # daytime images always have higher values than 10 for all three channels
            # values less than 10 are usually night
            # skip the image if it is night
            # if orange_value < 10 and red_value < 10: # higher than 250 for all means it is a white imag
            #     continue
            # # if the values are all higher than 250 then it is a white image and we want to remove it
            # if orange_value > 250 and red_value > 250:
            #     os.remove(file)
            #     print("Removed white image")
            #     continue
            # # if the image was not taken in the last x hours, skip it
            # if not is_recent(file, 60): # 60 minutes
            #     continue

            # blue_value = np.mean(image[:,:,1]) # blue value
            # print(orange_value, red_value, blue_value)
            # show the image and annotate it with the orange, red, and blue values
            # plt.imshow(image)
            # plt.title("Orange: " + str(orange_value) + " Red: " + str(red_value) + " Blue: " + str(blue_value))
            # plt.show()

            # if we reached this point the image can be added to the tapestry unless the tapestry has already been filled then just keep going without adding the image
            if (
                rotating
            ):  # if the tapestry is rotating, we take an image and add it to the tapestry as well as remove the oldest image otherwise we just add the image to the tapestry
                if sunsets_found == 10:
                    # remove the top image from the tapestry
                    # get the image at the top of the tapestry which has a height of total_height/10
                    top_image = blank_image[0:height, 0:width]
                    # crop the image to remove the top 1/10th of the image
                    blank_image = blank_image[height : height * 10, 0:width]
                    # add the new image to the bottom of the tapestry
                    blank_image = np.concatenate((blank_image, image), axis=0)
                    cv2.imwrite("images/tapestry.png", blank_image)
                else:
                    blank_image[
                        sunsets_found * height : (sunsets_found + 1) * height, 0:width
                    ] = image
                    # show progress by printing out the blank image
                    cv2.imwrite("images/tapestry.png", blank_image)
                    # print("Sunset found!")
                    sunsets_found += 1  # increment the number of sunsets found

            else:
                blank_image[
                    sunsets_found * height : (sunsets_found + 1) * height, 0:width
                ] = image
                # show progress by printing out the blank image
                cv2.imwrite("images/tapestry.png", blank_image)
                # print("Sunset found!")
                sunsets_found += 1  # increment the number of sunsets found
        except:
            print("Error reading image")
            print("line 386")
            pass


def stitched_panoramas(panel1, panel2, panel3, panel4, panel5, panel6):
    # get the image size
    img_width, img_height = panel1.shape[1], panel1.shape[0]
    # get the ratio of the width to height
    r = float(img_width) / float(img_height)
    # get the aspect ratio of the image
    ar = round(r, 2)
    # calculate the rotation angle
    rot = math.degrees(math.atan2(-panel1.get_top(), -panel1.get_left()))
    # get the rotation matrix for this angle
    m = cv2.getRotationMatrix2D((img_width / 2, img_height / 2), rot, 1)
    # multiply the two matrices together to get the final transformation
    new_im = cv2.warpAffine(panel1, m, (img_width, img_height))
    # find the top-left corner of the image
    pos = np.array((new_im.shape.width / 2, new_im.shape.height / 2))
    # crop the image to the correct size
    new_im = new_im.copy()
    #!cropped_im = new_im.crop(pos)
    # rotate the image back
    rotated_im = cv2.rotate(new_im, 90, pos, 0, 0, 360)
    # resize the image to the right size
    resized_im = cv2.resize(
        rotated_im, (int(round(ar * img_width)), int(round(ar * img_height)))
    )
    return resized_im


def get_image_size(image):
    """
    get_image_size returns the width and height of an image

    _extended_summary_

    :param image: the image to get the size of
    :type image: cv2 image
    :return: the width and height of the image
    :rtype: tuple
    """
    # get the image width and height
    w, h = image.shape[:2]
    # I am getting Exception has occurred: ValueError
    # too many values to unpack (expected 2)
    # the way to fix this is...
    # w, h = image.shape[:2]

    # get the ratio of the width to height
    r = float(w) / float(h)
    # get the aspect ratio of the image
    ar = round(r, 2)
    # calculate the rotation angle
    rot = math.degrees(math.atan2(-image.get_top(), -image.get_left()))
    # get the rotation matrix for this angle
    m = cv2.getRotationMatrix2D((w / 2, h / 2), rot, 1)
    # multiply the two matrices together to get the final transformation
    new_im = cv2.warpAffine(image, m, (w, h))
    # find the top-left corner of the image
    pos = np.array((new_im.shape.width / 2, new_im.shape.height / 2))
    # crop the image to the correct size
    new_im = new_im.crop(pos)
    # rotate the image back
    rotated_im = cv2.rotate(new_im, 90, pos, 0, 0, 360)
    # resize the image to the right size
    resized_im = cv2.resize(rotated_im, (int(round(ar * w)), int(round(ar * h))))
    return resized_im


# Implementing the trained h5 model in real-time using OpenCV and Python to detect scene elements or types in panel images from the cameras on the buoys.
# The model is already trained and saved in the models/buoy_model/keras_model.h5 file.

# # import the necessary packages
# from imutils import paths
# import numpy as np
# import cv2

# # import the load_model function from keras
# from keras.models import load_model

# # load the model from disk
# print("[INFO] loading model...")
# model = load_model('models/buoy_model/keras_model.h5')
# Labels for this model are shown below
# 0 Direct Sun
# 1 Stormy Weather
# 2 Interesting
# 3 Object Detected
# 4 Sunset
# 5 Clouds
# 6 Night


# def check_image(image):
#     global model # load the model from disk
#     # load the image and show it
#     image = cv2.imread('images/panels/44020/2022_11_6_10_54/panel_1.jpg')
#     cv2.imshow("image", image)
#     cv2.waitKey(0) # show image
#     # feed the image to the model and get the prediction
#     prediction = model.predict(image) # predict the image
#     print(prediction) # print the prediction

#     # Return the predicted class of the image (0-6)
#     return np.argmax(prediction) # return the predicted class i.e. the type of scene in the image 'Stormy Weather' or 'Direct Sun' etc.
#     # We return the np.argmax of the prediction because the prediction is a list of probabilities for each class. The class with the highest probability is the predicted class.


# # Do a test run of the check_image function to see if it works on a single image
# test_image = 'images/panels/42002/2022_11_6_18_30/panel_3.png'
# validation_class = 'Sunset' # we know that this image is a sunset image, so we can use this to test the model
# # we can use this to test the model

# # Testing Model:
# print("Testing Keras Model")
# assert(os.path.exists(test_image)) # check if file exists
# pred = check_image(test_image) # get the predicted class of the image
# if pred != 4:
#     print("The image is not a sunset image")
#     print(pred) # print the predicted class
# else:
#     print("The image is a sunset image")


def buoy_links():
    global ids
    links = [
        "https://www.ndbc.noaa.gov/buoycam.php?station=42001",
        "https://www.ndbc.noaa.gov/buoycam.php?station=46059",
        "https://www.ndbc.noaa.gov/buoycam.php?station=41044",
        "https://www.ndbc.noaa.gov/buoycam.php?station=46071",
        "https://www.ndbc.noaa.gov/buoycam.php?station=42002",
        "https://www.ndbc.noaa.gov/buoycam.php?station=46072",
        "https://www.ndbc.noaa.gov/buoycam.php?station=46066",
        "https://www.ndbc.noaa.gov/buoycam.php?station=41046",
        "https://www.ndbc.noaa.gov/buoycam.php?station=46088",
        "https://www.ndbc.noaa.gov/buoycam.php?station=44066",
        "https://www.ndbc.noaa.gov/buoycam.php?station=46089",
        "https://www.ndbc.noaa.gov/buoycam.php?station=41043",
        "https://www.ndbc.noaa.gov/buoycam.php?station=42012",
        "https://www.ndbc.noaa.gov/buoycam.php?station=42039",
        "https://www.ndbc.noaa.gov/buoycam.php?station=46012",
        "https://www.ndbc.noaa.gov/buoycam.php?station=46011",
        "https://www.ndbc.noaa.gov/buoycam.php?station=42060",
        "https://www.ndbc.noaa.gov/buoycam.php?station=41009",
        "https://www.ndbc.noaa.gov/buoycam.php?station=46028",
        "https://www.ndbc.noaa.gov/buoycam.php?station=44011",
        "https://www.ndbc.noaa.gov/buoycam.php?station=41008",
        "https://www.ndbc.noaa.gov/buoycam.php?station=46015",
        "https://www.ndbc.noaa.gov/buoycam.php?station=42059",
        "https://www.ndbc.noaa.gov/buoycam.php?station=44013",
        "https://www.ndbc.noaa.gov/buoycam.php?station=44007",
        "https://www.ndbc.noaa.gov/buoycam.php?station=46002",
        "https://www.ndbc.noaa.gov/buoycam.php?station=51003",
        "https://www.ndbc.noaa.gov/buoycam.php?station=46027",
        "https://www.ndbc.noaa.gov/buoycam.php?station=46026",
        "https://www.ndbc.noaa.gov/buoycam.php?station=51002",
        "https://www.ndbc.noaa.gov/buoycam.php?station=51000",
        "https://www.ndbc.noaa.gov/buoycam.php?station=42040",
        "https://www.ndbc.noaa.gov/buoycam.php?station=44020",
        "https://www.ndbc.noaa.gov/buoycam.php?station=46025",
        "https://www.ndbc.noaa.gov/buoycam.php?station=41010",
        "https://www.ndbc.noaa.gov/buoycam.php?station=41004",
        "https://www.ndbc.noaa.gov/buoycam.php?station=51001",
        "https://www.ndbc.noaa.gov/buoycam.php?station=44025",
        "https://www.ndbc.noaa.gov/buoycam.php?station=41001",
        "https://www.ndbc.noaa.gov/buoycam.php?station=51004",
        "https://www.ndbc.noaa.gov/buoycam.php?station=44027",
        "https://www.ndbc.noaa.gov/buoycam.php?station=41002",
        "https://www.ndbc.noaa.gov/buoycam.php?station=42020",
        "https://www.ndbc.noaa.gov/buoycam.php?station=46078",
        "https://www.ndbc.noaa.gov/buoycam.php?station=46087",
        "https://www.ndbc.noaa.gov/buoycam.php?station=51101",
        "https://www.ndbc.noaa.gov/buoycam.php?station=46086",
        "https://www.ndbc.noaa.gov/buoycam.php?station=45002",
        "https://www.ndbc.noaa.gov/buoycam.php?station=46053",
        "https://www.ndbc.noaa.gov/buoycam.php?station=46047",
        "https://www.ndbc.noaa.gov/buoycam.php?station=46084",
        "https://www.ndbc.noaa.gov/buoycam.php?station=46085",
        "https://www.ndbc.noaa.gov/buoycam.php?station=45003",
        "https://www.ndbc.noaa.gov/buoycam.php?station=45007",
        "https://www.ndbc.noaa.gov/buoycam.php?station=46042",
        "https://www.ndbc.noaa.gov/buoycam.php?station=45012",
        "https://www.ndbc.noaa.gov/buoycam.php?station=42019",
        "https://www.ndbc.noaa.gov/buoycam.php?station=46069",
        "https://www.ndbc.noaa.gov/buoycam.php?station=46054",
        "https://www.ndbc.noaa.gov/buoycam.php?station=41049",
        "https://www.ndbc.noaa.gov/buoycam.php?station=45005",
    ]

    # note: undo this to go with the established buoy list
    # links_2 = create_buoy_links(ids)

    # # append the links_2 to links if they are not already in links
    # for link in links_2:
    #     if link not in links:
    #         links.append(link)

    return links


# ids from cam_backend


def create_buoy_links(ids):
    # for each id in ids, create a link
    links = []
    for id in ids:
        link = "https://www.ndbc.noaa.gov/buoycam.php?station=" + id
        links.append(link)
    return links


# Notes:
# Buoy 42002 Has good sunsets


def check_buoy_image_ifwhite(image):
    """
    check_buoy_image_ifwhite checks if the image is white

    This function checks if the image is white. If the image is white, then the image is not valid and should be deleted.

    :param image: the image to check
    :type image: result of requests library get request for image url
    :return: True if the image is white, False if the image is not white
    :rtype: bool
    """
    # some buoys do not have a camera or the camera is not working. In these cases the image is white with only the text "No Image Available"
    # determine if the image is white
    # get the image in numpy array format
    img = np.asarray(bytearray(image.content), dtype="uint8")
    # convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # check if the image is white
    if np.mean(gray) > 250:
        return True
    else:
        return False


def ocean_stitching(imagePaths, pano_path):
    images = []
    # loop over the image paths, load each one, and add them to our
    # images to stitch list
    # open each image with cv2 and append to images list
    for imagePath in imagePaths:
        try:
            # add the full path to the image
            # '/Volumes/Backups of Grahams IMAC/PythonProjects/PySeas_Master_Folder/PySeas/images/panels/44020/2022_11_6_10_54/panel_1.jpg'
            # read the image
            full_path = (
                "/Volumes/Backups of Grahams IMAC/PythonProjects/PySeas_Master_Folder/PySeas/"
                + imagePath
            )
            imagePath = full_path  # this is the full path to the image
            assert os.path.exists(imagePath)  # check if file exists
            image = cv2.imread(imagePath, cv2.IMREAD_COLOR)  # read image
            # cv2.imshow("image", image)
            # cv2.waitKey(0) # show image
            images.append(image)  # append to list
            if image is None:
                print("Error loading image: " + imagePath)
        except:
            print("Error reading image: " + imagePath)
            continue
    # getting error: OpenCV(4.6.0) /Users/runner/work/opencv-python/opencv-python/opencv/modules/imgproc/src/resize.cpp:4052: error: (-215:Assertion failed) !ssize.empty() in function 'resize'
    # how to fix this:
    # The error is saying that the image is empty. This is because you are trying to read an image that doesn't exist. Check the path to the image and make sure it is correct.

    # initialize OpenCV's image stitcher object and then perform the image
    # stitching
    print("[INFO] stitching images...")
    stitcher = cv2.createStitcher() if imutils.is_cv3() else cv2.Stitcher_create()
    (status, stitched) = stitcher.stitch(images)
    # if the status is '0', then OpenCV successfully performed image
    # stitching
    if status == 0:
        # write the output stitched image to disk
        cv2.imwrite(pano_path, stitched)  # save the stitched image
        # display the output stitched image to our screen
        #!cv2.imshow("Stitched", stitched)
        #!cv2.waitKey(0)
    # otherwise the stitching failed, likely due to not enough keypoints)
    # being detected
    else:
        print("[INFO] image stitching failed ({})".format(status))


def refine_view(stitched_image):
    stitched = cv2.copyMakeBorder(
        stitched, 10, 10, 10, 10, cv2.BORDER_CONSTANT, (0, 0, 0)
    )
    # convert the stitched image to grayscale and threshold it
    # such that all pixels greater than zero are set to 255
    # (foreground) while all others remain 0 (background)
    gray = cv2.cvtColor(stitched, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]
    # find all external contours in the threshold image then find
    # the *largest* contour which will be the contour/outline of
    # the stitched image
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    # allocate memory for the mask which will contain the
    # rectangular bounding box of the stitched image region
    mask = np.zeros(thresh.shape, dtype="uint8")
    (x, y, w, h) = cv2.boundingRect(c)
    cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)


# given a list of images, create a pano image
def create_pano_image(image_list, pano_path):
    # create a pano image from the images in image_list
    # image_list is a list of image paths
    # pano_path is the path to save the pano image
    # create the pano image
    ocean_stitching(image_list, pano_path)
    # refine the pano image
    # refine_view(pano_path)


def chunk_images(buoy_id, foldername):
    """
    chunk_images takes a folder of images and splits them into sets of 6 images

    _extended_summary_

    :param buoy_id: The id of the buoy
    :type buoy_id: int
    :param foldername: The name of the folder containing the images
    :type foldername: str
    #"""
    # #buoy_id = str(foldername)
    # # There are many images in the folder. We want to parse these in sets of six.
    # # images/panels/41002/2022_11_5_15_44_panel_1.png is an example of the naming convention for the images.
    # # the first number (folder under panels) is the buoy id, then the filename begins with the year, the second number is the month, the third number is the day, the fourth number is the hour, the fifth number is the minute, and the sixth number is the panel number.
    # # we want to select the images that have the same buoy id, year, month, day, and hour AND end in the panel numbers: 1,2,3,4,5,6.

    # # for the example file above 2022_11_5_15_44_panel_1.png
    # # we want to put it into a subfolder named 2022_11_5_15 and put all the images that match the naming convention into that folder. There should only be 6.
    # # then return the list of lists of images that are in the subfolders.

    # # if the name of the

    # # foldername[0] --> '2022_11_5_15_44.jpg_panel_1.png'
    # # this is the kind of dirty names we have to clean.

    # # we want to get the first 4 numbers and the last number.
    # # we want to put the images into a folder named 2022_11_5_15
    # # we want to return a list of lists of images that are in the subfolders.

    # # we want to get the first 4 numbers and the last number.
    # # if type(foldername) == list:
    # #     foldername = foldername[0]
    # # else:
    # #     foldername = foldername
    # for file_name in tqdm(foldername):

    #     first_four = file_name.split('_')[0:4] # i.e. ['2022', '11', '5', '15']
    #     last_num = file_name.split('_')[-1] # i.e. '1.png'
    #     last_num = last_num.split('.')[0] # i.e. '1'

    #     # we want to put the images into a folder named 2022_11_5_15
    #     new_folder_name = '_'.join(first_four) # i.e. '2022_11_5_15'

    #     # the photos that belong with the new folder name (2022_11_5_15) contain the foldername's numbers.
    #     # Sort the images in the folder by the date they were added to the folder.
    #     #sorted_foldername = sorted(foldername, key=lambda x: os.path.getmtime(x)) # sort the images by the date they were added to the folder.

    #     # find all the files with the first 4 numbers (2022_11_5_15) and .jpg only once in the name
    #     files_with_last_four = [x for x in foldername if new_folder_name in x and 'jpg_panel' not in x]
    #     # find all the files with the last number (1)
    #     files_with_last_num = [x for x in files_with_last_four if str('panel_')+last_num in x]

    #     # are the images in the foldername in the correct order?
    #     # we want to check if the images are in the correct order.
    #     # They should now have the word panel 1, in the first image, panel 2 in the second image, etc. down to panel 6 and then the next image should be panel 1 again.
    #     # now we can just take each six images and put them into a folder.

    #     # logic
    #     # as long as the image contains "panel_n" where n is the next number in the order of the panels we add the image to the folder name that matches the first four numbers in their name.

    #      # go through the images in the foldername and put them into the correct folder.
    #     # we want to put the images into a folder named 2022_11_5_15 or whatever the first four numbers are.

    #     # make a folder in the panels folder with the name of the first four numbers.
    #     # if the folder doesn't exist, make it.
    #     print('new_folder_name',new_folder_name)
    #     try:
    #         os.mkdir(f'images/panels/{buoy_id}/batches/{new_folder_name}')
    #     except Exception as e:
    #         print(e)
    #         pass

    #     # go through the images in the foldername and put them into the correct folder.
    #     # we want to put the images into a folder named 2022_11_5_15 or whatever the first four numbers are.
    #     for image in files_with_last_num:
    #         # move the image into the directory f'images/panels/{buoy_id}/{new_folder_name}'
    #         # full path of the image
    #         full_path = f'images/panels/{buoy_id}/{image}'
    #         shutil.move(full_path, f'images/panels/{buoy_id}/batches/{new_folder_name}')
    #         pass

    #     # now we want to delete the images that are in the foldername but not in the subfolder.
    #     # we want to delete the images that are in the foldername but not in the subfolder.

    #     # get the list of images in the foldername
    #     images_in_foldername = [x for x in foldername if buoy_id in x]
    #     # get the list of images in the subfolder
    #     images_in_subfolder = [x for x in os.listdir(f'images/panels/{buoy_id}/batches/{new_folder_name}') if buoy_id in x]
    #     # get the list of images that are in the foldername but not in the subfolder
    #     #!images_to_delete = [x for x in images_in_foldername if x not in images_in_subfolder]
    #     # delete the images that are in the foldername but not in the subfolder
    #     #for image in images_to_delete:
    #     #    os.remove(f'images/panels/{buoy_id}/{image}')

    #     # img1 = os.path.join('images','panels',str(buoy_id),file_name) # the image we want to move
    #     # img2 = os.path.join('images','batches',str(buoy_id),new_folder_name) # the folder we want to move the image to.
    #     # # i.e. move the image from images/panels/41002/2022_11_5_15_44_panel_1.png to images/panels/41002/2022_11_5_15/2022_11_5_15_44_panel_1.png
    #     # shutil.copy(img1, img2) # copy the image to the folder.
    pass


# 51101,51000,51003,51002,51004 - Hawaii
# 46006 - Amazing Sunset
# 46089 - Tillamook Oregon.

from difPy import dif

cam_urls = buoy_links()  # get the links to the cameras
all_buoy_urls = create_buoy_links(ids)
stitch_switch = False  # make false if you don't want to stitch the images.

# open the blacklist file

from ratelimit import sleep_and_retry


# @limits(calls=1, period=4) # limit the number of calls to the function to 1 every 4 seconds.
@sleep_and_retry
def pull_data(cam_url, buoy_id, now):
    img = requests.get(cam_url)  # get the image
    if img.status_code == 200:
        return img
    else:
        print("status code", img.status_code, "for buoy", buoy_id)
    return img


### Testing the code
# detect red in an image
# load the image
image = cv2.imread("images/buoys/46072/2022_11_5_19_27.jpg")
image_path = "images/buoys/46072/2022_11_5_19_27.jpg"
# * Test 2.
# result = finding_red_version_two(image) # find the red in the image
# print(result)

# * Test 3. hsv and npwhere
output_img = finding_red_version_three(image_path)  # find the red in the image
print(output_img)
# print(output_hsv)


# Notes to self: remove functions for tests up to this point.
# * Test 4. Just red percent
# & Successful!
percent_red = detect_red_v4(image_path)
print(percent_red)


# test with the function to see if it detects red.

detect_red_v4(image_path)  # returns True if it detects red, False if it doesn't.


do_loop = True

if do_loop:
    pass
else:
    exit()  # exit the program if do_loop is False.


while True:
    try:
        # turn on at 4 am CST and turn off at 11 pm CST
        if (
            datetime.datetime.now().hour < 3 or datetime.datetime.now().hour > 24
        ):  # if it is before 3 am or after 12 am
            # wait to turn on until 4 am CST
            # keep the computer awake
            print("The computer is sleeping")
            time.sleep(240)  # sleep for 4 minutes
            continue

        # updated blacklist file
        blacklist = (
            open("data/blacklisted_buoy_ids.csv").read().splitlines()
        )  # get the list of buoy ids that are blacklisted.
        # parse blacklist to remove extra ' and " characters
        blacklist = [x.replace('"', "") for x in blacklist]
        blacklist = [x.replace("'", "") for x in blacklist]
        # create a blacklist list of strings from blacklist
        blacklist = [str(x) for x in blacklist][0].replace(" ", "").split(",")

        # # if the time is between 4 am and 11 am pacific time, then your wait_period is 100 seconds
        # if datetime.datetime.now().hour >= 4 and datetime.datetime.now().hour < 11:
        #     wait_period = 100
        # # if the time is between 11 am and 11 pm pacific time, then your wait_period is 600 seconds
        # if datetime.datetime.now().hour >= 11 and datetime.datetime.now().hour < 13:
        #     wait_period = 600 # 10 minutes
        # wait for 15 minutes
        wait_period = 600  # 10 minutes
        start_time = (
            datetime.datetime.now()
        )  # use this to calculate the next time to download images (every ten minutes)
        #!print('Starting the download loop at {}'.format(start_time))
        # print('I can still see things! Downloading images...')
        chunk_size = 30  # download 30 images at a time then pause for 10 seconds.
        chunk_size_current = (
            0  # the current number of images downloaded in the current chunk.
        )
        for cam_url in tqdm(cam_urls):
            # get the buoy id from the camera url
            buoy_id = re.search("station=(.*)", cam_url).group(1)
            if (
                buoy_id in blacklist
            ):  # if the buoy id is in the blacklist, then skip it.
                continue  # skip this buoy id
            # get the current time
            now = datetime.datetime.now()
            # create a directory for the buoy id if it doesn't already exist
            if not os.path.exists("images/buoys/{}".format(buoy_id)):
                os.makedirs("images/buoys/{}".format(buoy_id))
                ##logging.info("Created directory for buoy {}".format(buoy_id))

            # get the image
            ##logging.info("Checking buoy {}".format(buoy_id)) # log the buoy id
            if "images/buoys/{}/{}_{}_{}_{}_{}.jpg".format(
                buoy_id, now.year, now.month, now.day, now.hour, now.minute
            ) not in os.listdir(
                "images/buoys/{}".format(buoy_id)
            ):  # if the image has not already been downloaded
                time.sleep(
                    0.15
                )  # wait 0.25 seconds to avoid getting blocked by the server
                if (
                    chunk_size_current < chunk_size
                ):  # if we have not downloaded 30 images yet
                    chunk_size_current += 1  # add one to the chunk size
                else:
                    time.sleep(15)  # wait 15 seconds
                    chunk_size_current = 0  # reset the chunk size

                wait = True  # set the wait variable to true
                while wait:  # while we are waiting
                    try:  # try to get the image
                        img = pull_data(cam_url, buoy_id, now)  # download the image
                        wait = False
                    except Exception as e:
                        # print(e)
                        wait = True
                        time.sleep(1)
                        continue

                # check if the image is white

                # Print the name of the image we are downloading
                print(
                    "Downloading image: {}".format(
                        "images/buoys/{}/{}_{}_{}_{}_{}.jpg".format(
                            buoy_id, now.year, now.month, now.day, now.hour, now.minute
                        )
                    )
                )
                # save the image
                with open(
                    "images/buoys/{}/{}_{}_{}_{}_{}.jpg".format(
                        buoy_id, now.year, now.month, now.day, now.hour, now.minute
                    ),
                    "wb+",
                ) as f:
                    f.write(img.content)  # write the image to the file
                # check if the image is daytime or nighttime
                # ##logging.WARNING("Skipped night detection model for buoy {}".format(buoy_id))
                # if not is_it_daytime('images/buoys/{}/{}_{}_{}_{}_{}.jpg'.format(buoy_id, now.year, now.month, now.day, now.hour, now.minute)): # if it is nighttime
                # then we will delete the image
                # *print(f'Deleting image for buoy {buoy_id} because it is nighttime.')
                # *os.remove('images/buoys/{}/{}_{}_{}_{}_{}.jpg'.format(buoy_id, now.year, now.month, now.day, now.hour, now.minute))
                #    pass

                # ^ check image to see if it is just a white screen or not. If it is then we want to add this buoy id to the blacklist so that we don't download images from it anymore.

            else:
                print(
                    "Image already exists: {}".format(
                        "images/buoys/{}/{}_{}_{}_{}_{}.jpg".format(
                            buoy_id, now.year, now.month, now.day, now.hour, now.minute
                        )
                    )
                )
                pass  # if the image already exists, don't download it again

        ##logging.INFO("Beginning to panel images (line 24)") #! at {}".format(datetime.datetime.now()))
        # Save the panels to the images/panels directory
        list_of_buoys = os.listdir(
            "images/buoys"
        )  # get the list of buoy ids by their directory names

        # sample a random 20 extras from the
        print("Creating panels...")
        for buoy_id in tqdm(list_of_buoys):
            # get the list of images for the buoy
            # print(f'Paneling images for buoy {buoy_id}')
            if (
                buoy_id != ".DS_Store" and "." not in buoy_id
            ):  # if the buoy id is not a hidden file
                images = os.listdir("images/buoys/{}".format(buoy_id))
                # if the image has not already been used to create panels, create the panels and save them to the images/panels directory
                ##logging.info("Saving panels for buoy {}".format(buoy_id))
                for image in images:
                    # print(f'    Paneling image {image}')
                    # if the image is not None
                    if image == ".DS_Store" or image != "None":
                        continue
                    # If the panels directory for the buoy doesn't exist, create it.
                    if not os.path.exists("images/panels/{}".format(buoy_id)):
                        os.makedirs("images/panels/{}".format(buoy_id))
                    if (
                        "images/buoys/{}/{}".format(buoy_id, image)
                        in os.listdir("images/panels/{}".format(buoy_id))
                        and image == ".DS_Store"
                        and buoy_id != ".DS_Store"
                    ):
                        print(
                            "This image has already been used to create panels. Or it is a hidden file."
                        )
                    else:
                        # get the panels
                        (
                            panel_1,
                            panel_2,
                            panel_3,
                            panel_4,
                            panel_5,
                            panel_6,
                        ) = divide_into_panels(
                            buoy_id, "images/buoys/{}/{}".format(buoy_id, image)
                        )

                    # print('Processing image: {}'.format(image))

                    ##logging.info("Saved panels for buoy {}".format(buoy_id))
                    # print('Saving panels...')
                    # save the panels to the images/panels directory
                    # now, stitch these images together (correcting for the misalignment of the cameras) and save the result to the images/panoramas directory
                    # print('Stitching panels...')
                    # stitch the panels together using stitched_panoramas

                    # try:
                    #     stitched = stitched_panoramas(panel_1, panel_2, panel_3, panel_4, panel_5, panel_6)
                    #     # save the stitched panorama to the images/panoramas directory
                    #     # print('Saving stitched panorama...')
                    #     if not os.path.exists('images/panoramas/{}'.format(buoy_id)):
                    #         os.makedirs('images/panoramas/{}'.format(buoy_id))
                    #     # print('images/panoramas/{}/{}_{}_{}_{}_{}.jpg'.format(buoy_id, now.year, now.month, now.day, now.hour, now.minute))
                    #     cv2.imwrite('images/panoramas/{}/{}_{}_{}_{}_{}.jpg'.format(buoy_id, now.year, now.month, now.day, now.hour, now.minute), stitched)
                    #     # move the image to the images/panels directory
                    #     os.rename('images/buoys/{}/{}'.format(buoy_id, image), 'images/panels/{}/{}'.format(buoy_id, image))
                    #     ##logging.info("Moved image to panels directory for buoy {}".format(buoy_id))
                    # except Exception as e:
                    #     print(e)
                    #     pass

        # Stage 4: save buoy_update_rates_dict to a csv file
        buoy_update_rates_dict_df = pd.DataFrame.from_dict(
            buoy_update_rates_dict, orient="index"
        )
        buoy_update_rates_dict_df.to_csv("data/buoy_update_rates_dict.csv")

        # Stage 5: Using DifPy, find any images that are similar 'normal' to white_blank.jpg and delete them.
        # parse the buoy folders and their images

        try:
            buoy_folders = os.listdir("images/buoys")
            for buoy_folder in buoy_folders:
                if buoy_folder != ".DS_Store":
                    images = os.listdir("images/buoys/{}".format(buoy_folder))
                    for image in images:
                        if image != ".DS_Store" and image != "None":
                            # get the image path
                            image_path = "images/buoys/{}/{}".format(buoy_folder, image)
                            # get the image
                            image = cv2.imread(image_path)
                            white_image = cv2.imread("images/white_blank.jpg")
                            # we need these images to be the same size, so we will resize the white image to the size of the image
                            white_image = cv2.resize(
                                white_image, (image.shape[1], image.shape[0])
                            )
                            # are they ndarrays?
                            # print(type(image))
                            # print(type(white_image))

                            # get the difference between the image and the white_blank.jpg image
                            # calculate the difference between pixel values of the image and a pure white image using numpy
                            diff = np.sum(
                                np.abs(image - white_image)
                            )  # get the sum of the absolute difference between the two images
                            # if the difference is less than 1000, then we will delete the image
                            if diff < 1000:
                                print("Deleting image: {}".format(image_path))
                                # move the images instead of literally deleting them to a folder called 'deleted_images' in the images directory
                                if not os.path.exists("images/deleted_images"):
                                    os.makedirs("images/deleted_images")
                                os.rename(
                                    image_path,
                                    "images/deleted_images/{}_{}".format(
                                        image_path.split("/")[-1].split(".")[0],
                                        buoy_folder,
                                    ),
                                )
                                os.remove(image_path)
                            # get the difference score from the difference image
                            # difference_score = dif.get_difference_score()
                            # if the difference score is less than 0.1, then delete the image

                            # if the difference is less than 0.1, then delete the image
                            # if difference_score < 0.1:
                            #     os.remove(image_path)
                            #     print('Deleted image {} because it was too similar to white_blank.jpg'.format(image_path))
        except Exception as e:
            print("Error with White Image Detection: {}".format(e))
            pass

        if exper_1:
            # run DifPy on the images in the images/panels directory
            try:
                buoy_folders = os.listdir("images/panels")
                for buoy_folder in buoy_folders:
                    if buoy_folder != ".DS_Store":
                        images = os.listdir("images/panels/{}".format(buoy_folder))
                        for image in images:
                            if image != ".DS_Store":
                                # get the image path
                                image_path = "images/panels/{}/{}".format(
                                    buoy_folder, image
                                )
                                # get the image
                                image = cv2.imread(image_path)
                                white_image = cv2.imread("images/white_blank.jpg")
                                # we need these images to be the same size, so we will resize the white image to the size of the image
                                white_image = cv2.resize(
                                    white_image, (image.shape[1], image.shape[0])
                                )
                                # are they ndarrays?
                                # print(type(image))
                                # print(type(white_image))

                                # get the difference between the image and the white_blank.jpg image
                                # calculate the difference between pixel values of the image and a pure white image using numpy
                                diff = np.sum(
                                    np.abs(image - white_image)
                                )  # get the sum of the absolute difference between the two images
                                # if the difference is less than 1000, then we will delete the image
                                if diff < 1000:
                                    print("Deleting image: {}".format(image_path))
                                    # move the images instead of literally deleting them to a folder called 'deleted_images' in the images directory
                                    if not os.path.exists("images/deleted_images"):
                                        os.makedirs("images/deleted_images")
                                    os.rename(
                                        image_path,
                                        "images/deleted_images/{}_{}".format(
                                            image_path.split("/")[-1].split(".")[0],
                                            buoy_folder,
                                        ),
                                    )
                                    # os.remove(image_path)
                                # dif.
                                # # get the difference score from the difference image
                                # difference_score = dif.get_difference_score()
                                # # if the difference score is less than 0.1, then delete the image

                                # # if the difference is less than 0.1, then delete the image
                                # if difference_score < 0.1:
                                #     os.remove(image_path)
                                #     print('Deleted image {} because it was too similar to white_blank.jpg'.format(image_path))
            except Exception as e:
                print("Error with White Image Detection: {}".format(e))
                pass

        # Remove duplicate images (preferably before paneling but for now after)
        if duplicate_removal_flag == True:

            for folder in os.listdir("images/buoys"):
                if folder == ".DS_Store":
                    continue
                # get the list of images in the folder
                # sort the images by date
                # make folder_path variable from relative path
                folder_path = "images/buoys/{}".format(folder)
                search = dif(
                    folder_path,
                    similarity="high",
                    show_output=False,
                    show_progress=True,
                )  # returns a list of lists of similar images
                # for each list of similar images, move all but the first image to the deleted_images folder
                file_results_dict = search.result  # get the list of file names
                # {20220824212437767808 : {"filename" : "image1.jpg",
                #                         "location" : "C:/Path/to/Image/image1.jpg"},
                #                         "duplicates" : ["C:/Path/to/Image/duplicate_image1.jpg",
                #                                         "C:/Path/to/Image/duplicate_image2.jpg"]},
                # This is the format of the dictionary returned by the dif.search() method
                # I want to the filename, location, and duplicates
                # I want to move the duplicates to the deleted_images folder

                # make the deleted_images folder if it doesn't exist
                if not os.path.exists("images/deleted_images"):
                    os.makedirs("images/deleted_images")

                # counter should be how many files are in the deleted folder before we start
                counter = len(os.listdir("images/deleted_images"))
                # move the duplicates to the deleted_images folder
                for (
                    key
                ) in file_results_dict:  # iterate through the keys in the dictionary
                    # get the duplicates
                    value = file_results_dict[key]
                    duplicates = value["duplicates"]
                    for duplicate in duplicates:
                        try:
                            # move the duplicate to the deleted_images folder
                            # os.rename(duplicate, 'images/deleted_images/{}_{}'.format(counter,duplicate.split('/')[-1]))
                            # remove the duplicate
                            # full dupe path
                            # full_dupe_path = 'images/buoys/{}/{}'.format(folder, duplicate.split('/')[-1])

                            # first add "duplicate_" to the beginning of the file name
                            new_name = duplicate.split("/")[-1]  # get the file name
                            new_name = "duplicate_{}".format(
                                new_name
                            )  # add duplicate_ to the beginning of the file name
                            # then rename it in the same directory as the original
                            os.rename(
                                duplicate, "images/buoys/{}/{}".format(folder, new_name)
                            )
                            # then move the file to the deleted_images folder
                            print("Renamed {} to {}".format(duplicate, new_name))
                            # os.rename(duplicate, str(duplicate).replace('images/buoys', 'images/deleted_images'))
                            counter += 1
                        except Exception as e:
                            print("Error moving duplicate image: {}".format(e))
                            pass

        ignoring_panel_optimimal = True  # note: this is a temporary fix to the problem of the panel images not being generated
        # final step: make sure that all the previous buoy images have been panelled and saved to the images/panels directory
        for folder in tqdm(os.listdir("images/buoys")):
            # print('Checking if all images have been panelled for buoy {}'.format(folder))
            try:
                if folder == ".DS_Store":
                    continue
                images = os.listdir("images/buoys/{}".format(folder))
                # if the folder is not in the images/panels directory, then we need to panel the images
                # if not os.path.exists('images/panels/{}'.format(folder)):
                if ignoring_panel_optimimal:
                    # print('The images in the images/buoys/{} directory have not been panelled yet.'.format(folder))
                    # panelling the images
                    try:
                        os.mkdir("images/panels/{}".format(folder))
                        print(
                            "made directory for buoy {}".format(folder)
                            + " in images/panels"
                        )
                    except:
                        pass
                    batch_id = 1
                    for image in images:
                        # make a folder for the batch that has the same name as the image without the extension
                        try:
                            i_name = image[:-4]
                            directory_save_path = f"images/panels/{folder}/{i_name}"  # make the directory path
                            os.mkdir(directory_save_path)
                        except FileExistsError:
                            pass
                        # get the panels
                        # if the folder is not empty skip it
                        if len(os.listdir(directory_save_path)) > 0:
                            continue
                        try:
                            if image == ".DS_Store":
                                continue
                            # get the panels and save them to directory_save_path
                            (
                                panel_1,
                                panel_2,
                                panel_3,
                                panel_4,
                                panel_5,
                                panel_6,
                            ) = divide_into_panels(
                                folder, "images/buoys/{}/{}".format(folder, image)
                            )
                            # save the panels to the directory_save_path
                            # panel_1.save(f'{directory_save_path}/panel_1.jpg')
                            # panel_2.save(f'{directory_save_path}/panel_2.jpg')
                            # panel_3.save(f'{directory_save_path}/panel_3.jpg')
                            # panel_4.save(f'{directory_save_path}/panel_4.jpg')
                            # panel_5.save(f'{directory_save_path}/panel_5.jpg')
                            # panel_6.save(f'{directory_save_path}/panel_6.jpg')
                            ##logging.info('Saved panels for image: {}'.format(image))

                        except:
                            ##logging.ERROR('Error while saving panels for image: {}'.format(image))
                            # print('Could not create panels for image: {}'.format(image))
                            continue

                        # note: trying to add in the vincent code here
                        # stitch the images together
                        if stitch_switch:
                            files_to_stitch = [
                                f"{directory_save_path}/panel_1.jpg",
                                f"{directory_save_path}/panel_2.jpg",
                                f"{directory_save_path}/panel_3.jpg",
                                f"{directory_save_path}/panel_4.jpg",
                                f"{directory_save_path}/panel_5.jpg",
                                f"{directory_save_path}/panel_6.jpg",
                            ]  # list of files to stitch

                            # Stitch the images together with OpenCV and save the stitched image to the panoramas directory
                            print("Stitching images...")
                            try:
                                ocean_stitching(
                                    files_to_stitch,
                                    f"images/panoramas/{folder}/{i_name}.jpg",
                                )  # stitch the images together and save the stitched image to the panoramas directory
                            except Exception as f:
                                print(f)
                                print(
                                    "Could not stitch images for image: {}".format(
                                        image
                                    )
                                )
                            # > Overload resolution failed:
                            # >  - Can't parse 'images'. Sequence item with index 0 has a wrong type
                            # >  - Can't parse 'images'. Sequence item with index 0 has a wrong type
                            # >  - Stitcher.stitch() missing required argument 'masks' (pos 2)
                            # >  - Stitcher.stitch() missing required argument 'masks' (pos 2)
                            # fix: https://stackoverflow.com/questions/6380057/python-cv2-error-215-overload-resolution-failed

                        # try:
                        #     print('Stitching images for image set {}'.format(files_to_stitch))
                        #     panorama = vincent.make_panorama(images)
                        #     #remove the .jpg from the image name
                        #     image_name = image[:-4] # remove the .jpg from the image name

                        #     cv2.imwrite(f'images/panoramas/{folder}_{image_name}_panorama.png', panorama)
                        # except Exception as e:
                        #     print(e)
                        #     print('Could not stitch images together for buoy: {}'.format(folder))
                        #     continue

                        batch_id += 1
            except Exception as e:
                print(e)
                # print('Could not create panels for buoy: {}'.format(folder))
                # print('The images in the images/buoys/{} directory have not been panelled yet.'.format(folder))
                # print('line 139') # line 139
                pass
        # for each folder in the images/panels folder, stitch the images together and save them to the images/panoramas folder with the same name as the folder + panorama.png

        # //: the for loop below does not account for the fact that there are multiple captures with 6 frames per capture. This means that the images will be stitched together incorrectly. This is a problem that needs to be fixed. Find a way to select only the sets of 6 images that go together to stitch together.

        print("stage 5 complete")
        # Stage 6: Create the buoy dataframes
        # if it has been ten minutes since the last time the data was fetched, fetch the data again
        if time.time() - last_time_fetched > 600 or first_run:
            latest_data = (
                get_latest_data()
            )  # get the latest data from the RSS feed (updates every 10 minutes)
            # save the master dataframe to a csv file
            run_date = time.strftime("%Y%m%d_%H%M%S")
            latest_data.to_csv(f"data/rss/rss_buoy_data_{run_date}.csv", index=False)
            print("Done with this run")
            time_last_fetched_rss = time.time()  # get the time of the last fetch
        print("stage 6 complete")

        # * ======= show the last buoy image captured in this run
        try:
            display_last_image(
                list_of_buoys
            )  # display the last image captured in this run
        except:
            pass
        # * ====== End Show Buoy Image Snippet

        print("Running White Elimiination and populating the Tapestry")
        deal_with_white_images_and_populate_tapestry()  # run the white elimination and populate the tapestry

        # how much time has passed since the start of the loop?
        time_elapsed = datetime.datetime.now() - start_time
        # wait until the time elapsed is 15 minutes from the start of the loop
        print("Waiting for the remainder of the minutes...")

        if verbose_wait:
            # wait_period = 100 # was 900 (15 minutes)
            for i in tqdm(range(wait_period - time_elapsed.seconds)):
                time.sleep(1)
            iteration_counter += 1
        else:
            print(
                "Waiting for the remaining {} seconds".format(
                    wait_period - time_elapsed.seconds
                )
            )
            time.sleep(wait_period - time_elapsed.seconds)
            iteration_counter += 1

    except Exception as e:
        print(e)
        print("Error occurred.")
        # how much time has passed since the start of the loop?
        time_elapsed = datetime.datetime.now() - start_time
        # * wait till the ten minute mark is reached.
        for i in tqdm(range(wait_period - time_elapsed.seconds)):
            time.sleep(1)
        iteration_counter += 1

    # # now we have all the images in the subfolders
    # # we want to return a list of lists of images that are in the subfolders

    # list_of_lists = []
    # for folder in os.listdir(f'images/batches/{buoy_id}'):
    #     try:
    #         # Get the list of images in the folder, and append it to the list of lists (these are the relative paths to the images)
    #         list_of_lists.append(os.listdir(f'images/batches/{buoy_id}/{folder}')) # this will be a list of 6 images
    #     except NotADirectoryError:
    #         # there is a file in the folder
    #         continue
    #     except FileNotFoundError:
    #         # there is a missing folder
    #         print(f'FileNotFoundError: {folder}')
    #         continue
    #     except Exception as e:
    #         print(e)
    #         continue
    # return list_of_lists # return a list of sets of images that have the same buoy id, year, month, day, hour, and panel numbers 1,2,3,4,5,6


# Next Steps

"""

The next steps are to:
Use the Keras model saved in the models folder to predict the content of each panel image as it is created, and save the predictions to a csv file.
These predictions will determine which buoys the model will follow closest and which buoys it will eventually ignore. This will be done by finding which of the buoy cameras record the most interesting content (i.e. the most interesting content will be the content that the model predicts as being the most interesting).



            # daytime images always have higher values than 10 for all three channels
            # values less than 10 are usually night
            # skip the image if it is night
            #if orange_value < 10 and red_value < 10: # if the image is night
            #    continue
            # if the values are all higher than 250 then it is a white image and we want to remove it
            #if orange_value > 250 and red_value > 250:
            #    os.remove(file)
            #    print("Removed white image")
            #    continue
            # interesting photos have high red values
            #if red_value > 147 or (orange_value > 147 and #red_value > 100): # if the image is interesting
            #    add_list.append(file)
            # if the image was not taken in the last x hours, skip it

"""

This code appears to be a Python script that downloads images from buoys, crops them, and checks the colors of each image. It looks like the main function is check_colors_of_six_panels, which takes an image as input and returns the number of panels in the image that meet certain color criteria.

One suggestion for refactoring this code is to improve its readability and modularity. You could do this by breaking up the code into smaller, well-named functions that each perform a specific task. For example, instead of having all the code in a single function, you could create separate functions for downloading the images, cropping them, and checking their colors. This would make the code easier to understand and maintain, and would also allow you to reuse these functions in other parts of your project.

Another suggestion is to use more descriptive and consistent variable names. For example, instead of using abbreviations like img_width and img_height, you could use more descriptive names like image_width and image_height. This would make it easier to understand what the variables represent and would also improve the readability of your code.

In general, refactoring code involves making changes to the structure and organization of the code to improve its readability, maintainability, and performance. The specific changes you make will depend on your goals and the challenges you are facing with your code. It can be helpful to take a step back and think about what you want your code to do, and then make changes that will help you achieve those goals.




Graham Waters
how did I get the images?

Based on the code you provided, it looks like the images are being downloaded from buoys using the cam_backend module. This module appears to contain functions that download the images and save them to a specified location on your computer. The code then uses the glob module to find the downloaded images and processes them using the other functions in the script.

You can learn more about how the images are being downloaded and where they are being saved by looking at the cam_backend module and the specific functions that are being used to download the images. It is also worth checking the documentation for the cam_backend module to see if it provides any information about how the images are being downloaded and what the API endpoint is.

In general, the images for this project are being downloaded using the API provided by the National Oceanographic Atmospheric Association (NOAA). This API allows you to access data from buoys around the world, including images taken by the buoys. You can learn more about the NOAA API and how to use it by visiting their website and reading the documentation.




Graham Waters
# PySeas

## Purpose: The PySeas repo is an art generation repo. It trains on images from NOAA buoys and generates panoramic images of the ocean using GANs and OpenCV.

## Background: The NOAA buoy network is a network of buoys that are deployed in the ocean to collect data on weather conditions and ocean currents. The buoys take images of the ocean every hour and upload them to the NOAA website. The images are available for download and can be used to train machine learning models to generate panoramic images of the ocean.

It sounds like PySeas is an interesting project that uses machine learning and computer vision techniques to generate panoramic images of the ocean using data from NOAA buoys. Using GANs and OpenCV to generate these images is a creative approach that can produce some beautiful and unique art.

In order to train the machine learning model, you will need to collect a dataset of images from the NOAA buoy network. You can do this by using the API provided by the NOAA to download the images from the buoys. Once you have collected a sufficient number of images, you can use them to train a GAN or other machine learning model to generate panoramic images of the ocean.

To improve the quality of the generated images, you may want to use image preprocessing techniques such as cropping, resizing, and color correction. You could also experiment with different types of GANs or other machine learning models to see which ones produce the best results. By carefully selecting and preprocessing your training data and choosing the right machine learning model, you can create some beautiful and unique art using PySeas.




Graham Waters
Thank you! I am pretty excited about it.

I'm glad to hear that! It sounds like you have a great idea for a project and I'm sure that with some hard work and creativity, you will be able to create some beautiful and unique art using PySeas. Best of luck with your project, and feel free to reach out if you have any questions or need any further help.




Graham Waters
Okay, so when you said "gather images" how many images would I need to gather to train a GAN?

The number of images you need to train a GAN or other machine learning model will depend on a number of factors, including the complexity of the task, the quality of the images, and the capacity of the model. In general, the more data you have, the better your model will be able to learn and the more accurate its predictions will be.

As a rough guideline, you may want to start with a dataset of at least a few thousand images and see how well your model performs. If the performance is not good enough, you can try increasing the size of the dataset or experimenting with different model architectures and training strategies to see if that improves the results.

It's also important to remember that the quality of the images in your dataset is just as important as the quantity. Make sure that the images you use for training are high quality, properly aligned, and free of noise or other artifacts. This will help your model learn more effectively and produce better results.




Graham Waters
what would you recommend for my project structure? USING OOP and Python3, with open source in mind. Also consider deployability as a package (in the future) on PyPi.

When designing the structure of a project, it's important to consider factors such as maintainability, scalability, and reusability. One way to achieve these goals is to use object-oriented programming (OOP) and follow some best practices for organizing your code.

Here are some recommendations for the project structure:

Create a top-level directory for your project and give it a descriptive name that reflects the purpose of the project. For example, you could name the directory PySeas or OceanGAN.

Within the top-level directory, create a src directory for your Python code. This is where you will put your Python modules, classes, and functions.

Within the src directory, create separate files for each of your classes and modules. For example, you could create a BuoyData class that manages the data from the NOAA buoys, and a PanoramaGenerator class that generates the panoramic images using GANs.

Within each of these files, define your classes and functions using clear, descriptive names that reflect their purpose and behavior. Use comments and docstrings to provide additional information and context for your code.

In the top-level directory, create a tests directory for your unit tests. This is where you will put your test cases and test functions that verify that your code is working correctly.

Within the tests directory, create separate files for each of your classes and modules, and write test cases that cover the different scenarios and edge cases for your code. Use assert statements to check that your code is producing the expected results.

In the top-level directory, create a setup.py file that specifies the dependencies and other metadata for your project. This will allow you to package your project as a Python module and distribute it on PyPI or other package managers.

By following these recommendations, you can create a well-organized and maintainable project structure that is suitable for deployment as a Python package. This will make it easier to collaborate with others on the project, and will also make your code more reusable and scalable.




Graham Waters
 in src/ should my files be named like ... PanoramaGenerator.py ?

Yes, naming your files after the classes or modules they contain is a good practice that can improve the readability and maintainability of your code. For example, if you have a PanoramaGenerator class that generates panoramic images using GANs, you could create a file named PanoramaGenerator.py and define the class in that file. This makes it clear which file contains the class and what the class does, and also allows you to easily find and edit the class if you need to.

Another benefit of naming your files after the classes or modules they contain is that it can help prevent naming conflicts. For example, if you have two classes with the same name but in different files, you can avoid confusion by giving the files different names, such as PanoramaGenerator1.py and PanoramaGenerator2.py. This can make it easier to keep track of your code and avoid errors.

In general, using clear and descriptive names for your files and classes can improve the readability and maintainability of your code. It can also make it easier to collaborate with others and distribute your code as a package.




Graham Waters
Okay I created PanoramaGenerator.py in src. I know I need some other classes, but I am unsure what they are.

When designing the classes for your project, it can be helpful to think about the different tasks that your code will need to perform and the data that it will need to manage. This can help you identify the classes and functions that you will need to create, and can also help you organize your code into logical modules and layers.

For the PySeas project, you will likely need at least a few different classes to manage the data from the NOAA buoys, generate the panoramic images using GANs, and handle other tasks such as image preprocessing and color correction. Here are some examples of classes that you may want to consider creating:

A BuoyData class that manages the data from the NOAA buoys, including the images and other metadata. This class could provide methods for downloading the data from the NOAA API, storing it in a local database or file system, and accessing it for further processing.

A PanoramaGenerator class that generates the panoramic images using GANs. This class could provide methods for training the GAN on the buoy data, generating images using the trained model, and applying image preprocessing and color correction techniques to improve the quality of the generated images.

A ImageProcessor class that handles image preprocessing and color correction tasks. This class could provide methods for resizing, cropping, and adjusting the colors of the images to improve their quality and consistency.

A Classifier class that uses machine learning algorithms to classify the images into different categories, such as sunsets, storms, or calm waters. This class could provide methods for training a classifier on the buoy data, making predictions on new images, and evaluating the accuracy of the classifier.

These are just some examples of classes that you could create for the PySeas project. Depending on your specific goals and




Graham Waters
I started the BuoyData Class

"""
    class that manages the data from the NOAA buoys, including the images and other metadata. This class could provide methods for downloading the data from the NOAA API, storing it in a local database or file system, and accessing it for further processing.
"""

class BuoyData:
    def __init__(self, buoy_id, buoy_name, buoy_lat, buoy_lon, buoy_data):
        self.buoy_id = buoy_id
        self.buoy_name = buoy_name
        self.buoy_lat = buoy_lat
        self.buoy_lon = buoy_lon
        self.buoy_data = buoy_data

    def __str__(self):
        return "Buoy ID: " + self.buoy_id + "

That's a good start! Your BuoyData class provides a useful structure for managing the data from the NOAA buoys, and the __init__ and __str__ methods allow you to create and print instances of the class.

Here are some additional methods that you may want to consider adding to your BuoyData class:

A download_data method that uses the NOAA API to download the data for a particular buoy. This method could take the buoy ID as an input and use it to query the API and retrieve the data for that buoy.

A store_data method that saves the data for a particular buoy to a local database or file system. This method could take the buoy data as an input and use it to create a new entry in the database or save the data to a file.

A get_data method that retrieves the data for a particular buoy from the local database or file system. This method could take the buoy ID as an input and use it to query the database or read the data from the file, and then return the data for that buoy.

By implementing these methods, you can provide a more complete and functional BuoyData class that allows you to download, store, and access the data from the NOAA buoys. You can then use this class in your other code to manage the buoy data and use it for further processing and analysis.




Graham Waters
How does this look now?

"""
    class that manages the data from the NOAA buoys, including the images and other metadata. This class could provide methods for downloading the data from the NOAA API, storing it in a local database or file system, and accessing it for further processing.
"""

class BuoyData:
    """
        A download_data method that uses the NOAA API to download the data for a particular buoy. This method could take the buoy ID as an input and use it to query the API and retrieve the data for that buoy.

        A store_data method that saves the data for a particular buoy to a local database or file system. This method could take the buoy data as an input and use it to create a new entry in the database or save the data to a file.

        A get_data method that retrieves the data for a particular buoy from the local database or file system. This method could take the buoy ID as an input and use it to query the database or read the data from the file, and then return the data for that buoy.

    """
    def __init__(self, buoy_id, buoy_name, buoy_lat, buoy_lon, buoy_data):
        self.buoy_id = buoy_id
        self.buoy_name = buoy_name
        self.buoy_lat = buoy_lat
        self.buoy_lon = buoy_lon
        self.buoy_data = buoy_data

    def __str__(self):
        return "Buoy ID: " + self.buoy_id + "

    def download_data(self):
        pass

    def store_data(self):
        pass

    def get_data(self):
        pass

That looks better! Your updated BuoyData class now includes the download_data, store_data, and get_data methods that you described. These methods provide a useful framework for managing the data from the NOAA buoys, and you can now implement the specific code for each method to make them functional.

For example, in the download_data method, you can use the requests module to make an HTTP request to the NOAA API, passing the buoy ID as a parameter in the URL. You can then use the json module to parse the response and extract the data for that buoy. You can then save the data to a local file or database using the store_data method.

In the get_data method, you can use the buoy ID to query the local database or read the data from the file, and then return the data for that buoy. You can then use this method in your other code to access the data for a particular buoy and use it for further processing or analysis.

By implementing these methods in your BuoyData class, you can create a more complete and functional class that allows you to manage the data from the NOAA buoys and use it in your project. You can then use this class as a building block for your other classes and functions, and build a more powerful and flexible system for generating panoramic images of the ocean.




Graham Waters
Excellent, so what about the ImageProcessor class?

The ImageProcessor class could provide methods for performing image preprocessing and color correction tasks on the images from the NOAA buoys. This class could provide a number of different methods for adjusting the size, shape, and color of the images, and could be used to improve the quality and consistency of the images before they are used for training or generating panoramic images.

Here are some examples of methods that you may want to consider adding to your ImageProcessor class:

A resize_image method that resizes an image to a specified width and height. This method could take the image and the desired width and height as inputs, and then use the cv2 module (or another image processing library) to resize the image.

A crop_image method that crops an image to a specified rectangular region. This method could take the image and the coordinates of the top-left and bottom-right corners of the region to be cropped as inputs, and then use the cv2 module to crop the image.

A color_correct method that adjusts the colors of an image to improve its appearance. This method could take the image as an input, and then use the cv2 module to apply color correction techniques such as white balancing or color temperature adjustments to the image.

By implementing these methods in your ImageProcessor class, you can create a useful tool for preprocessing and color correcting the images from the NOAA buoys. You can then use this class in your other code to improve the quality of the images and make them more suitable for training or generating panoramic images.




Graham Waters
show me how that should look

Here is an example of how the ImageProcessor class could be implemented in Python:

Copy code
import cv2

class ImageProcessor:
    def __init__(self):
        pass

    def resize_image(self, image, width, height):
        # use cv2 to resize the image
        resized_image = cv2.resize(image, (width, height))
        return resized_image

    def crop_image(self, image, top_left, bottom_right):
        # use cv2 to crop the image
        cropped_image = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
        return cropped_image

    def color_correct(self, image):
        # use cv2 to apply color correction to the image
        color_corrected_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return color_corrected_image
In this example, the ImageProcessor class includes three methods: resize_image, crop_image, and color_correct. These methods use the cv2 module to perform the specified image processing tasks, and return the processed images as output.

The resize_image method takes an image and the desired width and height as inputs, and uses the cv2.resize function to resize the image to the specified dimensions. The crop_image method takes an image and the coordinates of the top-left and bottom-right corners of the region to be cropped, and uses array slicing to crop the image. The color_correct method takes an image as input and uses the cv2.cvtColor function to apply color correction to the image.

By implementing these methods in your ImageProcessor class, you can create a useful tool for preprocessing and color correcting the images from the NOAA buoys. You can then use this class in your other code to improve the quality of the images and make them more suitable for training or generating panoramic images.




Graham Waters
What about the Classifier Class?

The Classifier class could be used to train a machine learning model to classify the images from the NOAA buoys according to certain criteria, such as whether the image shows a sunset or not. This class could provide methods for training a model on a dataset of labeled images, evaluating the model on a test set of images, and making predictions on new images.

Here are some examples of methods that you may want to consider adding to your Classifier class:

A train method that trains a machine learning model on a dataset of labeled images. This method could take the dataset and the labels as inputs, and use a machine learning library (such as scikit-learn) to train a model on the dataset. The method could also take hyperparameters for the model (such as the type of model to use, the learning rate, etc.) as additional inputs.

A evaluate method that evaluates a trained model on a test set of images. This method could take the test set and the corresponding labels as inputs, and use the trained model to make predictions on the test set. The method could then compute various metrics (such as accuracy, precision, recall, etc.) to evaluate the performance of the model on the test set.

A predict method that makes predictions on new images using a trained model. This method could take the new images and the trained model as inputs, and use the model to make predictions on the images. The method could return the predicted labels for the images as output.

By implementing these methods in your Classifier class, you can create a useful tool for training and evaluating machine learning models that can classify the images from the NOAA buoys. You can then use this class in your other code to train a model




Graham Waters
can you show that to me as well?

Here is an example of how the Classifier class could be implemented in Python:

Copy code
from sklearn.svm import SVC

class Classifier:
    def __init__(self, model_type="SVM", learning_rate=0.1):
        # initialize the type of model to use and the learning rate
        self.model_type = model_type
        self.learning_rate = learning_rate

    def train(self, X, y):
        # if the model type is SVM, use an SVM classifier
        if self.model_type == "SVM":
            self.model = SVC(gamma="auto", learning_rate=self.learning_rate)
            self.model.fit(X, y)

    def evaluate(self, X, y):
        # use the trained model to make predictions on the test set
        y_pred = self.model.predict(X)
        # compute the accuracy of the predictions
        accuracy = sum(y == y_pred) / len(y)
        return accuracy

    def predict(self, X):
        # use the trained model to make predictions on the new images
        y_pred = self.model.predict(X)
        return y_pred
In this example, the Classifier class includes three methods: train, evaluate, and predict. The train method uses the SVC class from the sklearn library to train a support vector machine (SVM) classifier on a dataset of labeled images. The evaluate method uses the trained model to make predictions on a test set of images, and computes the accuracy of the predictions. The predict method uses the trained model to make predictions on new images.

By implementing these methods in your Classifier class, you can create a useful tool for training and evaluating machine learning models that can classify the images from the NOAA buoys. You can then use this class in your other code to train a model and use it to classify new images according to your chosen criteria.




Graham Waters
Okay, now what about PanoramaGenerator? Is that where we finally use GANs?

Yes, the PanoramaGenerator class could be used to generate panoramic images of the ocean using a generative adversarial network (GAN). This class could provide methods for training a GAN on a dataset of images from the NOAA buoys, and using the trained GAN to generate new panoramic images of the ocean.

Here are some examples of methods that you may want to consider adding to your PanoramaGenerator class:

A train method that trains a GAN on a dataset of images from the NOAA buoys. This method could take the dataset as an input, and use a GAN library (such as TensorFlow or PyTorch) to train the GAN on the dataset. The method could also take hyperparameters for the GAN (such as the size of the latent space, the number of training iterations, etc.) as additional inputs.

A generate method that uses a trained GAN to generate new panoramic images of the ocean. This method could take the trained GAN and the desired size of the generated image as inputs, and use the GAN to generate a new panoramic image of the ocean. The method could return the generated image as output.

A save method that saves a generated image to a local file. This method could take the generated image and a file name as inputs, and use an image library (such as OpenCV or Pillow) to save the generated image to a file with the specified name.

By implementing these methods in your PanoramaGenerator class, you can create a useful tool for training and using GANs to generate panoramic images of the ocean. You can then use this class in your other code to train a GAN and use it to generate new panoramic images of the ocean.




Graham Waters
show that to me in python3

Here is an example of how the PanoramaGenerator class could be implemented in Python using the TensorFlow library:

Copy code
import tensorflow as tf

class PanoramaGenerator:
    def __init__(self):
        # initialize the TensorFlow session
        self.sess = tf.Session()

    def train(self, dataset, latent_space_size, num_iterations):
        # define the inputs to the GAN
        input_data = tf.placeholder(tf.float32, shape=(None, dataset.shape[1]))
        input_noise = tf.placeholder(tf.float32, shape=(None, latent_space_size))
        # define the generator network
        generator = self.build_generator(input_noise)
        # define the discriminator network
        discriminator = self.build_discriminator(input_data)
        # define the GAN loss function
        loss = self.gan_loss(generator, discriminator, input_data, input_noise)
        # define the optimizer for the GAN
        optimizer = tf.train.AdamOptimizer()
        # define the training operation for the GAN
        train_op = optimizer.minimize(loss)
        # initialize the TensorFlow variables
        self.sess.run(tf.global_variables_initializer())
        # train the GAN for the specified number of iterations
        for i in range(num_iterations):
            # sample random noise vectors for the generator input
            noise = np.random.randn(batch_size, latent_space_size)
            # sample random images from the dataset for the discriminator input
            batch = dataset.sample(batch_size)
            # run a training step for the GAN
            _, loss_val = self.sess.run([train_op, loss], feed_dict={input_data: batch, input_noise: noise})
            # print the loss value every 100 iterations
            if i % 100 == 0:
                print("Iteration %d: Loss = %.4f" % (i, loss_val))

    def generate(self, noise):
        # generate a panoramic image using the GAN
        generator = self.build_generator(noise)
        panorama = self.sess.run(generator)
        return panorama

    def save(self, panorama, file_name):
        # save the panor



Graham Waters
show me the save method again please.

Sure, here is the save method from the PanoramaGenerator class example again:

Copy code
def save(self, panorama, file_name):
    # save the panoramic image to a file
    cv2.imwrite(file_name, panorama)
This method takes the generated panoramic image and the desired file name as inputs, and uses the OpenCV library to save the image to a file with the specified name. You can use this method to save the generated panoramic images to files for later use or analysis.




Graham Waters
how does all of this tie together?

Here is an example of how all of the classes could work together to generate panoramic images from the NOAA buoy data:

Copy code
# create a BuoyData instance to manage the buoy data
buoy_data = BuoyData()

# download the data for a particular buoy using the NOAA API
buoy_data.download_data("44011")

# store the downloaded data in a local database
buoy_data.store_data()

# retrieve the data for the buoy from the local database
data = buoy_data.get_data("44011")

# create an ImageProcessor instance to pre-process the images
image_processor = ImageProcessor()

# pre-process the images in the buoy data
data = image_processor.pre_process(data)

# create a Classifier instance to classify the images as sunsets or non-sunsets
classifier = Classifier()

# train the classifier on the pre-processed images
classifier.train(data)

# classify the images in the buoy data as sunsets or non-sunsets
data = classifier.classify(data)

# select the sunset images from the buoy data
sunset_images = data.filter(is_sunset=True)

# create a PanoramaGenerator instance to generate panoramic images from the sunset images
panorama_generator = PanoramaGenerator()

# train the panorama generator on the sunset images
panorama_generator.train(sunset_images, latent_space_size=100, num_iterations=10000)

# generate a panoramic image using the trained generator
noise = np.random.randn(100)
panorama = panorama_generator.generate(noise)

# save the generated panoramic image to a file
panorama_generator.save(panorama, "panorama.jpg")
This code creates instances of the BuoyData, ImageProcessor, Classifier, and PanoramaGenerator classes, and uses them to download and store buoy data, pre-process and classify the images, and train and generate panoramic images. Finally, it saves the generated panoramic image to a file.




Graham Waters
from pyseas.src.BuoyData import BuoyData
from pyseas.src.ImageProcessor import ImageProcessor
from pyseas.src.Classifier import Classifier
from src.ImageProcessor import ImageProcessor
from pyseas.src.PanoramaGenerator import PanoramaGenerator

# create a BuoyData instance to manage the buoy data
buoy_data = BuoyData()

# download the data for a particular buoy using the NOAA API
buoy_data.download_data("44011")

# store the downloaded data in a local database
buoy_data.store_data()

# retrieve the data for the buoy from the local database
data = buoy_data.get_data("44011")

# create an ImageProcessor instance to pre-process the images
image_processor = ImageProcessor()

# pre-process the images in the buoy data
data = image_processor.pre_process(data)

# create a Classifier instance to classify the images as sunsets or non-sunsets
classifier = Classifier()

# train the classifier on the pre-processed images
classifier.train(data)

# classify the images in the buoy data as sunsets or non-sunsets
data = classifier.classify(data)

# select the sunset images from the buoy data
sunset_images = data.filter(is_sunset=True)

# create a PanoramaGenerator instance to generate panoramic images from the sunset images
panorama_generator = PanoramaGenerator()

# train the panorama generator on the sunset images
panorama_generator.train(sunset_images, latent_space_size=100, num_iterations=10000)

# generate a panoramic image using the trained generator
noise = np.random.randn(100)
panorama = panorama_generator.generate(noise)

# save the generated panoramic image to a file
panorama_generator.save(panorama, "panorama.jpg")

Here is an example of how you could use the BuoyData, ImageProcessor, Classifier, and PanoramaGenerator classes in your PySeas project:

Copy code
from pyseas.src.BuoyData import BuoyData
from pyseas.src.ImageProcessor import ImageProcessor
from pyseas.src.Classifier import Classifier
from pyseas.src.PanoramaGenerator import PanoramaGenerator

# create a BuoyData instance to manage the buoy data
buoy_data = BuoyData()

# download the data for a particular buoy using the NOAA API
buoy_data.download_data("44011")

# store the downloaded data in a local database
buoy_data.store_data()

# retrieve the data for the buoy from the local database
data = buoy_data.get_data("44011")

# create an ImageProcessor instance to pre-process the images
image_processor = ImageProcessor()

# pre-process the images in the buoy data
data = image_processor.pre_process(data)

# create a Classifier instance to classify the images as sunsets or non-sunsets
classifier = Classifier()

# train the classifier on the pre-processed images
classifier.train(data)

# classify the images in the buoy data as sunsets or non-sunsets
data = classifier.classify(data)

# select the sunset images from the buoy data
sunset_images = data.filter(is_sunset=True)

# create a PanoramaGenerator instance to generate panoramic images from the sunset images
panorama_generator = PanoramaGenerator()

# train the panorama generator on the sunset images



Graham Waters
ImportError: dlopen(/Users/grahamwaters/micromamba/envs/MASTER_ENV/lib/python3.9/lib-dynload/_ctypes.cpython-39-darwin.so, 0x0002): Library not loaded: @rpath/libffi.7.dylib
  Referenced from: <1473791D-F729-392A-A674-C6C0C3BA1B1E> /Users/grahamwaters/micromamba/envs/MASTER_ENV/lib/python3.9/lib-dynload/_ctypes.cpython-39-darwin.so
  Reason: tried: '/Users/grahamwaters/micromamba/envs/MASTER_ENV/lib/libffi.7.dylib' (mach-o file, but is an incompatible architecture (have 'x86_64', need 'arm64')), '/System/Volumes/Preboot/Cryptexes/OS/Users/grahamwaters/micromamba/envs/MASTER_ENV/lib/libffi.7.dylib' (no such file), '/Users/grahamwaters/micromamba/envs/MASTER_ENV/lib/libffi.7.dylib' (mach-o file, but is an incompatible architecture (have 'x86_64', need 'arm64')), '/System/Volumes/Preboot/Cryptexes/OS/Users/grahamwaters/micromamba/envs/MASTER_ENV/lib/libffi.7.dylib' (no such file), '/Users/grahamwaters/micromamba/envs/MASTER_ENV/lib/python3.9/lib-dynload/../../libffi.7.dylib' (mach-o file, but is an incompatible architecture (have 'x86_64', need 'arm64')), '/Users/grahamwaters/micromamba/envs/MASTER_ENV/lib/libffi.7.dylib' (mach-o file, but is an incompatible architecture (have 'x86_64', need 'arm64')), '/System/Volumes/Preboot/Cryptexes/OS/Users/grahamwaters/micromamba/envs/MASTER_ENV/lib/libffi.7.dylib' (no such file), '/Users/grahamwaters/micromamba/envs/MASTER_ENV/lib/libffi.7.dylib' (mach-o file, but is an incompatible architecture (have 'x86_64', need 'arm64')), '/System/Volumes/Preboot/Cryptexes/OS/Users/grahamwaters/micromamba/envs/MASTER_ENV/lib/libffi.7.dylib' (no such file), '/Users/grahamwaters/micromamba/envs/MASTER_ENV/lib/python3.9/lib-dynload/../../libffi.7.dylib' (mach-o file, but is an incompatible architecture (have 'x86_64', need 'arm64')), '/Users/grahamwaters/micromamba/envs/MASTER_ENV/lib/libffi.7.dylib' (mach-o file, but is an incompatible architecture (have 'x86_64', need 'arm64')), '/System/Volumes/Preboot/Cryptexes/OS/Users/grahamwaters/micromamba/envs/MASTER_ENV/lib/libffi.7.dylib' (no such file), '/Users/grahamwaters/micromamba/envs/MASTER_ENV/bin/../lib/libffi.7.dylib' (mach-o file, but is an incompatible architecture (have 'x86_64', need 'arm64')), '/Users/grahamwaters/micromamba/envs/MASTER_ENV/lib/libffi.7.dylib' (mach-o file, but is an incompatible architecture (have 'x86_64', need 'arm64')), '/System/Volumes/Preboot/Cryptexes/OS/Users/grahamwaters/micromamba/envs/MASTER_ENV/lib/libffi.7.dylib' (no such file), '/Users/grahamwaters/micromamba/envs/MASTER_ENV/bin/../lib/libffi.7.dylib' (mach-o file, but is an incompatible architecture (have 'x86_64', need 'arm64')), '/System/Volumes/Preboot/Cryptexes/OS@rpath/libffi.7.dylib' (no such file), '/Users/grahamwaters/micromamba/envs/MASTER_ENV/lib/libffi.7.dylib' (mach-o file, but is an incompatible architecture (have 'x86_64', need 'arm64')), '/System/Volumes/Preboot/Cryptexes/OS/Users/grahamwaters/micromamba/envs/MASTER_ENV/lib/libffi.7.dylib' (no such file), '/Users/grahamwaters/micromamba/envs/MASTER_ENV/lib/libffi.7.dylib' (mach-o file, but is an incompatible architecture (have 'x86_64', need 'arm64')), '/System/Volumes/Preboot/Cryptexes/OS/Users/grahamwaters/micromamba/envs/MASTER_ENV/lib/libffi.7.dylib' (no such file), '/Users/grahamwaters/micromamba/envs/MASTER_ENV/lib/python3.9/lib-dynload/../../libffi.7.dylib' (mach-o file, but is an incompatible architecture (have 'x86_64', need 'arm64')), '/Users/grahamwaters/micromamba/envs/MASTER_ENV/lib/libffi.7.dylib' (mach-o file, but is an incompatible architecture (have 'x86_64', need 'arm64')), '/System/Volumes/Preboot/Cryptexes/OS/Users/grahamwaters/micromamba/envs/MASTER_ENV/lib/libffi.7.dylib' (no such file), '/Users/grahamwaters/micromamba/envs/MASTER_ENV/lib/libffi.7.dylib' (mach-o file, but is an incompatible architecture (have 'x86_64', need 'arm64')), '/System/Volumes/Preboot/Cryptexes/OS/Users/grahamwaters/micromamba/envs/MASTER_ENV/lib/libffi.7.dylib' (no such file), '/Users/grahamwaters/micromamba/envs/MASTER_ENV/lib/python3.9/lib-dynload/../../libffi.7.dylib' (mach-o file, but is an incompatible architecture (have 'x86_64', need 'arm64')), '/Users/grahamwaters/micromamba/envs/MASTER_ENV/lib/libffi.7.dylib' (mach-o file, but is an incompatible architecture (have 'x86_64', need 'arm64')), '/System/Volumes/Preboot/Cryptexes/OS/Users/grahamwaters/micromamba/envs/MASTER_ENV/lib/libffi.7.dylib' (no such file), '/Users/grahamwaters/micromamba/envs/MASTER_ENV/bin/../lib/libffi.7.dylib' (mach-o file, but is an incompatible architecture (have 'x86_64', need 'arm64')), '/Users/grahamwaters/micromamba/envs/MASTER_ENV/lib/libffi.7.dylib' (mach-o file, but is an incompatible architecture (have 'x86_64', need 'arm64')), '/System/Volumes/Preboot/Cryptexes/OS/Users/grahamwaters/micromamba/envs/MASTER_ENV/lib/libffi.7.dylib' (no such file), '/Users/grahamwaters/micromamba/envs/MASTER_ENV/bin/../lib/libffi.7.dylib' (mach-o file, but is an incompatible architecture (have 'x86_64', need 'arm64')), '/usr/local/lib/libffi.7.dylib' (no such file), '/usr/lib/libffi.7.dylib' (no such file, not in dyld cache)

It looks like there is an issue with the architecture of the libffi library. This library is required by the ctypes module, which is used by some of the code in your project. The error message indicates that the library is compiled for x86_64 architecture, but your system is expecting a library compiled for arm64 architecture.

To fix this issue, you can try installing a version of the libffi library that is compiled for arm64 architecture. You can do this using your system's package manager, or by building the library from source. Alternatively, you can try using a different version of Python that is compiled for the same architecture as the libffi library.




Graham Waters
can I do it with conda?

Yes, you can use conda to create a new environment and install the necessary packages for your project. Here is an example of how you might do that:

Open a terminal and run the following command to create a new environment called "pyseas" with Python 3.9 and the necessary packages for your project:
Copy code
conda create -n pyseas python=3.9 opencv numpy tensorflow
Activate the environment by running the following command:
Copy code
conda activate pyseas
Once the environment is activated, you can run your Python script by using the python command, followed by the name of your script, e.g.:
Copy code
python my_script.py
This will ensure that your script uses the correct version of Python and has access to the necessary packages. You can deactivate the environment at any time by running the conda deactivate command.




Graham Waters
Exception has occurred: SyntaxError
EOL while scanning string literal (BuoyData.py, line 22)
  File "/Users/grahamwaters/Library/CloudStorage/GoogleDrive-gewaters2@gmail.com/My Drive/PySeas/src/pyseas_main.py", line 1, in <module>
    from BuoyData import BuoyData

It looks like there is a problem with the syntax on line 22 of the BuoyData.py file. The error message says that there is an "EOL" (end of line) character while the interpreter was scanning a string literal. This means that there is probably an unterminated string somewhere on that line. Check to make sure that all of your string literals are properly quoted and terminated with a closing quotation mark.




Graham Waters
I fixed it and made this change

class BuoyData:
    """
        A download_data method that uses the NOAA API to download the data for a particular buoy. This method could take the buoy ID as an input and use it to query the API and retrieve the data for that buoy.

        A store_data method that saves the data for a particular buoy to a local database or file system. This method could take the buoy data as an input and use it to create a new entry in the database or save the data to a file.

        A get_data method that retrieves the data for a particular buoy from the local database or file system. This method could take the buoy ID as an input and use it to query the database or read the data from the file, and then return the data for that buoy.

    """
    def __init__(self, buoy_id, buoy_name, buoy_lat, buoy_lon, buoy_data):
        self.buoy_id = buoy_id
        self.buoy_name = buoy_name
        self.buoy_lat = buoy_lat
        self.buoy_lon = buoy_lon
        self.buoy_data = buoy_data

    def __str__(self):
        return "Buoy ID: " + self.buoy_id + " Buoy Name: " + self.buoy_name + " Buoy Latitude: " + self.buoy_lat + " Buoy Longitude: " + self.buoy_lon + " Buoy Data: " + self.buoy_data

    def download_data(self):
        pass

    def store_data(self):
        pass

    def get_data(self):
        pass

This looks good. Here's an example of how the classes you have defined could be used together to generate a panoramic image of the ocean using data from a NOAA buoy:

Copy code
# create a BuoyData instance to manage the buoy data
buoy_data = BuoyData()

# download the data for a particular buoy using the NOAA API
buoy_data.download_data("44011")

# store the downloaded data in a local database
buoy_data.store_data()

# retrieve the data for the buoy from the local database
data = buoy_data.get_data("44011")

# create an ImageProcessor instance to pre-process the images
image_processor = ImageProcessor()

# pre-process the images in the buoy data
data = image_processor.pre_process(data)

# create a Classifier instance to classify the images as sunsets or non-sunsets
classifier = Classifier()

# train the classifier on the pre-processed images
classifier.train(data)

# classify the images in the buoy data as sunsets or non-sunsets
data = classifier.classify(data)

# select the sunset images from the buoy data
sunset_images = data.filter(is_sunset=True)

# create a PanoramaGenerator instance to generate panoramic images from the sunset images
panorama_generator = PanoramaGenerator()

# train the panorama generator on the sunset images
panorama_generator.train(sunset_images, latent_space_size=100, num_iterations=10000)

# generate a panoramic image using the trained generator
noise = np.random.randn(100)
panorama = panorama_generator.generate(noise)

# save the generated panoramic image to a file
panorama_generator.save(panorama, "panorama.jpg")
In this code, we first create an instance of the BuoyData class to manage the data from the NOAA buoys. We then use the download_data method to download the data for a particular buoy from the NOAA API, and the store_data method to save the data to a local database.

Next, we use the get_data method to retrieve the data for the buoy from the local database, and pass it to an instance of the ImageProcessor class to pre-process the images.

Once the images have been pre-processed, we pass them to an instance of the Classifier class to classify the images as sunsets or non-sunsets. We then use the train method to train the classifier on the pre-processed images, and the classify method to classify the images in the buoy data.

Next, we use the filter method to select only the sunset images from the buoy data, and pass them to an instance of the PanoramaGenerator class to generate panoramic images. We use the train method to train the generator on the sunset images, and the generate method to generate a panoramic image. Finally, we use the save method to save the generated panoramic image to a file.




Graham Waters
I had to install sklearn

It looks like you are trying to use the scikit-learn library in your project. This library is a popular machine learning library for Python that provides a wide range of tools and algorithms for data analysis and machine learning. To install it, you can use the following command:

Copy code
pip install scikit-learn
Alternatively, if you are using Anaconda as your Python package manager, you can use the following command to install scikit-learn:

Copy code
conda install -c anaconda scikit-learn
Once you have installed the library, you can import it in your Python code using the following import statement:

Copy code
import sklearn
After importing the library, you can use its various tools and algorithms in your code. For example, if you want to use a support vector machine (SVM) classifier from the library, you can import it using the following import statement:

Copy code
from sklearn.svm import SVC
You can then create an instance of the SVC class and use it to train and evaluate your model. For more information on how to use scikit-learn, you can refer to the library's documentation: https://scikit-learn.org/stable/documentation.html.




Graham Waters
and import numpy as np in the main file as well.
Exception has occurred: TypeError
__init__() missing 5 required positional arguments: 'buoy_id', 'buoy_name', 'buoy_lat', 'buoy_lon', and 'buoy_data'
  File "/Users/grahamwaters/Library/CloudStorage/GoogleDrive-gewaters2@gmail.com/My Drive/PySeas/src/pyseas_main.py", line 8, in <module>
    buoy_data = BuoyData()

It looks like you forgot to pass the required parameters to the BuoyData constructor when you created the buoy_data instance. The __init__ method of the BuoyData class defines five required parameters: buoy_id, buoy_name, buoy_lat, buoy_lon, and buoy_data. You need to pass these parameters when you create a new instance of the BuoyData class. For example:

Copy code
# create a BuoyData instance to manage the buoy data
buoy_data = BuoyData("44011", "Buoy Name", "37.75", "-122.37", "Buoy Data")
This creates a new BuoyData instance and sets the buoy_id, buoy_name, buoy_lat, buoy_lon, and buoy_data attributes to the values specified in the arguments. You can then use this instance to call the other methods in the BuoyData class.

Make sure to pass the required parameters when you create any new instances of the BuoyData class.

To train a GAN for art generation using Ray, you would need to perform the following steps:

Install Ray and any necessary dependencies, such as TensorFlow and NumPy, using pip or another package manager.

Define the Generator and Discriminator classes, which will be used to build the GAN model. These classes should have methods for building the model, training the model, and generating output images.

Create an instance of the Ray Client, which will be used to manage the distributed training process.

Create an instance of the Generator class, and use its build method to define the architecture of the generator network. This should include defining the input and output layers, as well as any hidden layers and their activation functions.

Create an instance of the Discriminator class, and use its build method to define the architecture of the discriminator network. This should include defining the input and output layers, as well as any hidden layers and their activation functions.

Use the train method of the Generator and Discriminator classes to train the GAN model on a dataset of art images. This should involve using the Adam optimizer to minimize the loss function of the model, and periodically generating and saving output images to track the progress of the training. Use the Ray Client to distribute the training process across multiple CPUs or GPUs, as needed.

Use the generate method of the Generator class to generate output images using the trained GAN model. This can be done by passing in a random noise vector as input to the generator network, and using the resulting output image as the style for a target content image.

Save the trained GAN model, along with any generated output images, for future use or further analysis.


To train a GAN on images of the ocean, you will need to perform the following steps:

Collect and prepare the data: First, you need to collect a dataset of images of the ocean that you want to use for training. This typically involves downloading and organizing the photos into a structured format, such as a folder named ocean within the data/images directory. You may also need to preprocess the images to resize them or perform other transformations to make them suitable for input to the neural network.

Load the data: Next, you need to use TensorFlow or other libraries to load the images from the data/images/ocean directory. This typically involves reading the images from the file system and decoding them, and then applying any necessary preprocessing steps such as resizing or converting the images to grayscale. The resulting images should be returned as a NumPy array.

Split the dataset: Once the images have been loaded, you need to split the dataset into training and evaluation sets. This typically involves shuffling the images and using a specified ratio to determine the split point between the training and evaluation sets. The resulting sets should be returned as separate NumPy arrays.

Define the model: Next, you need to define the GAN model, including the generator and discriminator networks. This typically involves defining the model architecture and specifying the layers, activations, and other components of the networks. You can use TensorFlow or other libraries to define the model, and you can customize the model as needed to suit the specific requirements of your GAN.

Train the model: Once the GAN model has been defined, you can use TensorFlow or other libraries to train the model on the training set. This typically involves iteratively feeding the training images to the GAN, using the optimization algorithm to adjust the model weights and biases, and measuring the loss to evaluate the performance of the model. You can use the evaluation set to periodically evaluate the performance of the trained model.

Generate images: Once the GAN model has been trained, you can use it to generate new images of the ocean. This typically involves feeding random noise vectors to the generator network, and using the generated images as output. You can use the generated images for various purposes, such as visualizing the results of the GAN, or further refining the GAN model by fine-tuning the weights and biases.



Graham Waters
how do I train a neural network on a set of images.
To train a neural network on a set of images, you need to do the following:

Collect and prepare the data: The first step is to collect a dataset of images that you want to use for training. This typically involves downloading and organizing the images into a structured format, such as folders for each class of images. You may also need to preprocess the images to resize them or perform other transformations to make them suitable for input to the neural network.

Define the model architecture: Next, you need to define the architecture of the neural network that you will use for training. This typically involves deciding on the number and size of the layers in the network, as well as the specific types of layers (such as convolutional, pooling, or fully-connected layers) and the activation functions that will be used.

Train the model: Once the data and model architecture are defined, you can begin the training process. This typically involves feeding the data into the model in batches, using an optimization algorithm (such as stochastic gradient descent) to adjust the model's weights and biases in order to minimize the error between the predicted and true values.

Evaluate the model: After training, it's important to evaluate the performance of the model on a separate dataset that it hasn't seen before. This will give you an idea of how well the model generalizes to new data and can help you identify any overfitting or underfitting issues.

Fine-tune the model: Depending on the results of the evaluation, you may need to fine-tune the model by adjusting the architecture, hyperparameters, or other aspects of the training process in order to improve its performance. This can involve trying different architectures, changing the learning rate or other hyperparameters, or using techniques such as regularization or early stopping to prevent overfitting.

Overall, training a neural network on a set of images is a complex process that requires a thorough understanding of deep learning and a lot of experimentation. It's often helpful to use existing tools and libraries (such as TensorFlow or PyTorch) that can make the process of building and training a neural network easier.




Graham Waters
Collect and prepare the data: The first step is to collect a dataset of images you want to use for training. This typically involves downloading and organizing the photos into a structured format, such as folders for each class of images. You may also need to preprocess the images to resize them or perform other transformations to make them suitable for input to the neural network.
Yes, that's correct. Collecting and preparing the data is an important step in training a neural network on images. It involves gathering a dataset of images and organizing them into a structured format, such as folders for each class of images. Preprocessing the images can also be important, as it can help to improve the performance of the model by making the data more consistent and suitable for input to the neural network. Preprocessing steps can include resizing the images to a standard size, converting them to a specific format (such as grayscale), and performing other transformations to extract features or improve the quality of the data.




Graham Waters
I am writing the function to preprocess the images for training a GAN. What should I start with?
When preprocessing images for training a GAN, the first step is typically to load the images from their source (such as a folder on your computer) and convert them to a format that can be used as input to the GAN. This can involve reading the images into memory using an image processing library (such as OpenCV), and then converting them to a numerical representation (such as a NumPy array) that can be fed into the GAN.

Once the images are loaded and converted, you can then perform any necessary preprocessing steps, such as resizing the images to a standard size, normalizing the pixel values to a specific range (such as -1 to 1), and applying any other transformations that are needed to extract features or improve the quality of the data.

It's also important to make sure that the preprocessing steps are consistent with the requirements of the GAN, such as the input shape and data type expected by the generator and discriminator networks. This will ensure that the preprocessed images can be fed into the GAN without any issues.

Overall, preprocessing images for training a GAN involves a combination of loading and converting the images, performing necessary preprocessing steps, and ensuring that the preprocessed images are compatible with the GAN.




Graham Waters
show me a preprocessing method for this GAN
Here is an example of a preprocessing method that can be used to prepare images for training a GAN:

Copy code
def preprocess_images(images):
  # Convert the images from RGB to grayscale
  grayscale_images = tf.image.rgb_to_grayscale(images)

  # Resize the images to 64x64 pixels
  resized_images = tf.image.resize(grayscale_images, [64, 64])

  # Normalize the pixel values to the range [-1, 1]
  normalized_images = (resized_images - 127.5) / 127.5

  return normalized_images
This preprocessing method converts the images from RGB to grayscale, resizes them to 64x64 pixels, and normalizes the pixel values to the range -1 to 1. These steps are commonly used in GANs to reduce the complexity of the data and make it more suitable for training. You can modify or extend this preprocessing method as needed to suit the specific requirements of your GAN.




Graham Waters
Use this code as a basis for making our GAN:

import os, sys
sys.path.append(os.getcwd())

from random import randint

import time
import functools
import math

import numpy as np
import tensorflow as tf

import tflib as lib
import tflib.ops.linear
import tflib.ops.conv2d
import tflib.ops.batchnorm
import tflib.ops.deconv2d
import tflib.save_images
import tflib.wikiartGenre
import tflib.ops.layernorm
import tflib.plot


MODE = 'acwgan' # dcgan, wgan, wgan-gp, lsgan
DIM = 64 # Model dimensionality
CRITIC_ITERS = 5 # How many iterations to train the critic for
N_GPUS = 1 # Number of GPUs
BATCH_SIZE = 84 # Batch size. Must be a multiple of CLASSES and N_GPUS
ITERS = 200000 # How many iterations to train for
LAMBDA = 10 # Gradient penalty lambda hyperparameter
OUTPUT_DIM = 64*64*3 # Number of pixels in each iamge
CLASSES = 14 #Number of classes, for genres probably 14
PREITERATIONS = 2000 #Number of preiteration training cycles to run
lib.print_model_settings(locals().copy())

def GeneratorAndDiscriminator():
    return kACGANGenerator, kACGANDiscriminator


DEVICES = ['/gpu:{}'.format(i) for i in range(N_GPUS)]

def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha*x, x)

def ReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(name+'.Linear', n_in, n_out, inputs, initialization='he')
    return tf.nn.relu(output)

def LeakyReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(name+'.Linear', n_in, n_out, inputs, initialization='he')
    return LeakyReLU(output)

def Batchnorm(name, axes, inputs):

    if ('Discriminator' in name) and (MODE == 'wgan-gp' or MODE == 'acwgan'):
        if axes != [0,2,3]:
            raise Exception('Layernorm over non-standard axes is unsupported')
        return lib.ops.layernorm.Layernorm(name,[1,2,3],inputs)
    else:
        return lib.ops.batchnorm.Batchnorm(name,axes,inputs,fused=True)

def pixcnn_gated_nonlinearity(name, output_dim, a, b, c=None, d=None):
    if c is not None and d is not None:
        a = a + c
        b = b + d

    result = tf.sigmoid(a) * tf.tanh(b)
    return result

def SubpixelConv2D(*args, **kwargs):
    kwargs['output_dim'] = 4*kwargs['output_dim']
    output = lib.ops.conv2d.Conv2D(*args, **kwargs)
    output = tf.transpose(output, [0,2,3,1])
    output = tf.depth_to_space(output, 2)
    output = tf.transpose(output, [0,3,1,2])
    return output

def ResidualBlock(name, input_dim, output_dim, filter_size, inputs, resample=None, he_init=True):
    """
    resample: None, 'down', or 'up'
    """
    if resample=='down':
        conv_shortcut = functools.partial(lib.ops.conv2d.Conv2D, stride=2)
        conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=input_dim//2)
        conv_1b       = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim//2, output_dim=output_dim//2, stride=2)
        conv_2        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=output_dim//2, output_dim=output_dim)
    elif resample=='up':
        conv_shortcut = SubpixelConv2D
        conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=input_dim//2)
        conv_1b       = functools.partial(lib.ops.deconv2d.Deconv2D, input_dim=input_dim//2, output_dim=output_dim//2)
        conv_2        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=output_dim//2, output_dim=output_dim)
    elif resample==None:
        conv_shortcut = lib.ops.conv2d.Conv2D
        conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim,  output_dim=input_dim//2)
        conv_1b       = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim//2,  output_dim=output_dim//2)
        conv_2        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim//2, output_dim=output_dim)

    else:
        raise Exception('invalid resample value')

    if output_dim==input_dim and resample==None:
        shortcut = inputs # Identity skip-connection
    else:
        shortcut = conv_shortcut(name+'.Shortcut', input_dim=input_dim, output_dim=output_dim, filter_size=1,
                                 he_init=False, biases=True, inputs=inputs)

    output = inputs
    output = tf.nn.relu(output)
    output = conv_1(name+'.Conv1', filter_size=1, inputs=output, he_init=he_init, weightnorm=False)
    output = tf.nn.relu(output)
    output = conv_1b(name+'.Conv1B', filter_size=filter_size, inputs=output, he_init=he_init, weightnorm=False)
    output = tf.nn.relu(output)
    output = conv_2(name+'.Conv2', filter_size=1, inputs=output, he_init=he_init, weightnorm=False, biases=False)
    output = Batchnorm(name+'.BN', [0,2,3], output)

    return shortcut + (0.3*output)

# ! Generators

def kACGANGenerator(n_samples, numClasses, labels, noise=None, dim=DIM, bn=True, nonlinearity=tf.nn.relu, condition=None):
    lib.ops.conv2d.set_weights_stdev(0.02)
    lib.ops.deconv2d.set_weights_stdev(0.02)
    lib.ops.linear.set_weights_stdev(0.02)
    if noise is None:
        noise = tf.random_normal([n_samples, 128])

    labels = tf.cast(labels, tf.float32)
    noise = tf.concat([noise, labels], 1)

    output = lib.ops.linear.Linear('Generator.Input', 128+numClasses, 8*4*4*dim*2, noise) #probs need to recalculate dimensions
    output = tf.reshape(output, [-1, 8*dim*2, 4, 4])
    if bn:
        output = Batchnorm('Generator.BN1', [0,2,3], output)
    condition = lib.ops.linear.Linear('Generator.cond1', numClasses, 8*4*4*dim*2, labels,biases=False)
    condition = tf.reshape(condition, [-1, 8*dim*2, 4, 4])
    output = pixcnn_gated_nonlinearity('Generator.nl1', 8*dim, output[:,::2], output[:,1::2], condition[:,::2], condition[:,1::2])


    output = lib.ops.deconv2d.Deconv2D('Generator.2', 8*dim, 4*dim*2, 5, output)
    if bn:
        output = Batchnorm('Generator.BN2', [0,2,3], output)
    condition = lib.ops.linear.Linear('Generator.cond2', numClasses, 4*8*8*dim*2, labels)
    condition = tf.reshape(condition, [-1, 4*dim*2, 8, 8])
    output = pixcnn_gated_nonlinearity('Generator.nl2', 4*dim,output[:,::2], output[:,1::2], condition[:,::2], condition[:,1::2])

    output = lib.ops.deconv2d.Deconv2D('Generator.3', 4*dim, 2*dim*2, 5, output)
    if bn:
        output = Batchnorm('Generator.BN3', [0,2,3], output)
    condition = lib.ops.linear.Linear('Generator.cond3', numClasses, 2*16*16*dim*2, labels)
    condition = tf.reshape(condition, [-1, 2*dim*2, 16, 16])
    output = pixcnn_gated_nonlinearity('Generator.nl3', 2*dim,output[:,::2], output[:,1::2], condition[:,::2], condition[:,1::2])

    output = lib.ops.deconv2d.Deconv2D('Generator.4', 2*dim, dim*2, 5, output)
    if bn:
        output = Batchnorm('Generator.BN4', [0,2,3], output)
    condition = lib.ops.linear.Linear('Generator.cond4', numClasses, 32*32*dim*2, labels)
    condition = tf.reshape(condition, [-1, dim*2, 32, 32])
    output = pixcnn_gated_nonlinearity('Generator.nl4', dim, output[:,::2], output[:,1::2], condition[:,::2], condition[:,1::2])

    output = lib.ops.deconv2d.Deconv2D('Generator.5', dim, 3, 5, output)

    output = tf.tanh(output)

    lib.ops.conv2d.unset_weights_stdev()
    lib.ops.deconv2d.unset_weights_stdev()
    lib.ops.linear.unset_weights_stdev()

    return tf.reshape(output, [-1, OUTPUT_DIM]), labels

def kACGANDiscriminator(inputs, numClasses, dim=DIM, bn=True, nonlinearity=LeakyReLU):
    output = tf.reshape(inputs, [-1, 3, 64, 64])

    lib.ops.conv2d.set_weights_stdev(0.02)
    lib.ops.deconv2d.set_weights_stdev(0.02)
    lib.ops.linear.set_weights_stdev(0.02)

    output = lib.ops.conv2d.Conv2D('Discriminator.1', 3, dim, 5, output, stride=2)
    output = nonlinearity(output)

    output = lib.ops.conv2d.Conv2D('Discriminator.2', dim, 2*dim, 5, output, stride=2)
    if bn:
        output = Batchnorm('Discriminator.BN2', [0,2,3], output)
    output = nonlinearity(output)

    output = lib.ops.conv2d.Conv2D('Discriminator.3', 2*dim, 4*dim, 5, output, stride=2)
    if bn:
        output = Batchnorm('Discriminator.BN3', [0,2,3], output)
    output = nonlinearity(output)


    output = lib.ops.conv2d.Conv2D('Discriminator.4', 4*dim, 8*dim, 5, output, stride=2)
    if bn:
        output = Batchnorm('Discriminator.BN4', [0,2,3], output)
    output = nonlinearity(output)
    finalLayer = tf.reshape(output, [-1, 4*4*8*dim])

    sourceOutput = lib.ops.linear.Linear('Discriminator.sourceOutput', 4*4*8*dim, 1, finalLayer)

    classOutput = lib.ops.linear.Linear('Discriminator.classOutput', 4*4*8*dim, numClasses, finalLayer)

    lib.ops.conv2d.unset_weights_stdev()
    lib.ops.deconv2d.unset_weights_stdev()
    lib.ops.linear.unset_weights_stdev()



    return (tf.reshape(sourceOutput, [-1]), tf.reshape(classOutput, [-1, numClasses]))


def genRandomLabels(n_samples, numClasses,condition=None):
    labels = np.zeros([BATCH_SIZE,CLASSES], dtype=np.float32)
    for i in range(n_samples):
        if condition is not None:
            labelNum = condition
        else:
            labelNum = randint(0, numClasses-1)
        labels[i, labelNum] = 1
    return labels

Generator, Discriminator = GeneratorAndDiscriminator()

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as session:

    all_real_data_conv = tf.placeholder(tf.int32, shape=[BATCH_SIZE, 3, 64, 64])
    all_real_label_conv = tf.placeholder(tf.int32, shape=[BATCH_SIZE,CLASSES])

    generated_labels_conv = tf.placeholder(tf.int32, shape=[BATCH_SIZE,CLASSES])
    sample_labels_conv = tf.placeholder(tf.int32, shape=[BATCH_SIZE,CLASSES])

    if tf.__version__.startswith('1.'):
        split_real_data_conv = tf.split(all_real_data_conv, len(DEVICES))
        split_real_label_conv = tf.split(all_real_label_conv, len(DEVICES))
        split_generated_labels_conv = tf.split(generated_labels_conv, len(DEVICES))
        split_sample_labels_conv = tf.split(sample_labels_conv, len(DEVICES))
    else:
        split_real_data_conv = tf.split(0, len(DEVICES), all_real_data_conv)
        split_real_data_label = tf.split(0, len(DEVICES), all_real_data_conv)
        split_generated_labels = tf.split(0, len(DEVICES), generated_labels_conv)
        split_sample_labels = tf.split(0, len(DEVICES), sample_labels_conv)

    gen_costs, disc_costs = [],[]

    for device_index, (device, real_data_conv, real_label_conv) in enumerate(zip(DEVICES, split_real_data_conv, split_real_label_conv)):
        with tf.device(device):

            real_data = tf.reshape(2*((tf.cast(real_data_conv, tf.float32)/255.)-.5), [BATCH_SIZE//len(DEVICES), OUTPUT_DIM])
            real_labels = tf.reshape(real_label_conv, [BATCH_SIZE//len(DEVICES), CLASSES])

            generated_labels = tf.reshape(split_generated_labels_conv, [BATCH_SIZE//len(DEVICES), CLASSES])
            sample_labels = tf.reshape(split_sample_labels_conv, [BATCH_SIZE//len(DEVICES), CLASSES])

            fake_data, fake_labels= Generator(BATCH_SIZE//len(DEVICES), CLASSES, generated_labels)

            #set up discrimnator results

            disc_fake,disc_fake_class = Discriminator(fake_data, CLASSES)
            disc_real,disc_real_class = Discriminator(real_data, CLASSES)

            prediction = tf.argmax(disc_fake_class, 1)
            correct_answer = tf.argmax(fake_labels, 1)
            equality = tf.equal(prediction, correct_answer)
            genAccuracy = tf.reduce_mean(tf.cast(equality, tf.float32))

            prediction = tf.argmax(disc_real_class, 1)
            correct_answer = tf.argmax(real_labels, 1)
            equality = tf.equal(prediction, correct_answer)
            realAccuracy = tf.reduce_mean(tf.cast(equality, tf.float32))

            gen_cost = -tf.reduce_mean(disc_fake)
            disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

            gen_cost_test = -tf.reduce_mean(disc_fake)
            disc_cost_test = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

            generated_class_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=disc_fake_class,
                                                                                              labels=fake_labels))


            real_class_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=disc_real_class,
                                                                                              labels=real_labels))
            gen_cost += generated_class_cost
            disc_cost += real_class_cost

            alpha = tf.random_uniform(
                shape=[BATCH_SIZE//len(DEVICES),1],
                minval=0.,
                maxval=1.
            )
            differences = fake_data - real_data
            interpolates = real_data + (alpha*differences)
            gradients = tf.gradients(Discriminator(interpolates, CLASSES)[0], [interpolates])[0]
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
            gradient_penalty = tf.reduce_mean((slopes-1.)**2)
            disc_cost += LAMBDA*gradient_penalty

            real_class_cost_gradient = real_class_cost*50 + LAMBDA*gradient_penalty


            gen_costs.append(gen_cost)
            disc_costs.append(disc_cost)

    gen_cost = tf.add_n(gen_costs) / len(DEVICES)
    disc_cost = tf.add_n(disc_costs) / len(DEVICES)

    gen_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(gen_cost,
                                                                                             var_list=lib.params_with_name('Generator'),
                                                                                             colocate_gradients_with_ops=True)
    disc_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(disc_cost,
                                                                                              var_list=lib.params_with_name('Discriminator.'),
                                                                                              colocate_gradients_with_ops=True)
    class_train_op =  tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(real_class_cost_gradient,
                                                                                                var_list=lib.params_with_name('Discriminator.'),
                                                                                                colocate_gradients_with_ops=True)
    # For generating samples

    fixed_noise = tf.constant(np.random.normal(size=(BATCH_SIZE, 128)).astype('float32'))
    all_fixed_noise_samples = []
    for device_index, device in enumerate(DEVICES):
        n_samples = BATCH_SIZE // len(DEVICES)
        all_fixed_noise_samples.append(Generator(n_samples, CLASSES, sample_labels,noise=fixed_noise[device_index*n_samples:(device_index+1)*n_samples])[0])
        if tf.__version__.startswith('1.'):
            all_fixed_noise_samples = tf.concat(all_fixed_noise_samples, axis=0)
        else:
            all_fixed_noise_samples = tf.concat(0, all_fixed_noise_samples)


    def generate_image(iteration):
        for i in range(CLASSES):
            curLabel= genRandomLabels(BATCH_SIZE,CLASSES,condition=i)
            samples = session.run(all_fixed_noise_samples, feed_dict={sample_labels: curLabel})
            samples = ((samples+1.)*(255.99/2)).astype('int32')
            lib.save_images.save_images(samples.reshape((BATCH_SIZE, 3, 64, 64)), 'generated/samples_{}_{}.png'.format(str(i), iteration))



    # Dataset iterator
    train_gen, dev_gen = lib.wikiartGenre.load(BATCH_SIZE)

    def softmax_cross_entropy(logit, y):
        return -tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=y))

    def inf_train_gen():
        while True:
            for (images,labels) in train_gen():
                yield images,labels


    _sample_labels = genRandomLabels(BATCH_SIZE, CLASSES)
    # Save a batch of ground-truth samples
    _x,_y = next(train_gen())
    _x_r = session.run(real_data, feed_dict={all_real_data_conv: _x})
    _x_r = ((_x_r+1.)*(255.99/2)).astype('int32')
    lib.save_images.save_images(_x_r.reshape((BATCH_SIZE, 3, 64, 64)), 'generated/samples_groundtruth.png')



    session.run(tf.initialize_all_variables(), feed_dict={generated_labels_conv: genRandomLabels(BATCH_SIZE,CLASSES)})
    gen = train_gen()

    for iterp in range(PREITERATIONS):
        _data, _labels = next(gen)
        _ , accuracy = session.run([disc_train_op, realAccuracy],feed_dict = {all_real_data_conv: _data, all_real_label_conv: _labels, generated_labels_conv: genRandomLabels(BATCH_SIZE, CLASSES)})
        if iterp % 100 == 99:
            print('pretraining accuracy: ' + str(accuracy))


    for iteration in range(ITERS):
        start_time = time.time()
        # Train generator
        if iteration > 0:
            _ = session.run(gen_train_op, feed_dict={generated_labels_conv: genRandomLabels(BATCH_SIZE,CLASSES)})
        # Train critic
        disc_iters = CRITIC_ITERS
        for i in range(disc_iters):
            _data, _labels = next(gen)
            _disc_cost, _disc_cost_test, class_cost_test, gen_class_cost, _gen_cost_test, _genAccuracy, _realAccuracy, _ = session.run([disc_cost, disc_cost_test, real_class_cost, generated_class_cost, gen_cost_test, genAccuracy, realAccuracy, disc_train_op], feed_dict={all_real_data_conv: _data, all_real_label_conv: _labels, generated_labels_conv: genRandomLabels(BATCH_SIZE,CLASSES)})

        lib.plot.plot('train disc cost', _disc_cost)
        lib.plot.plot('time', time.time() - start_time)
        lib.plot.plot('wgan train disc cost', _disc_cost_test)
        lib.plot.plot('train class cost', class_cost_test)
        lib.plot.plot('generated class cost', gen_class_cost)
        lib.plot.plot('gen cost cost', _gen_cost_test)
        lib.plot.plot('gen accuracy', _genAccuracy)
        lib.plot.plot('real accuracy', _realAccuracy)

        if (iteration % 100 == 99 and iteration<1000) or iteration % 1000 == 999 :
            t = time.time()
            dev_disc_costs = []
            images, labels = next(dev_gen())
            _dev_disc_cost, _dev_disc_cost_test, _class_cost_test, _gen_class_cost, _dev_gen_cost_test, _dev_genAccuracy, _dev_realAccuracy = session.run([disc_cost, disc_cost_test, real_class_cost, generated_class_cost, gen_cost_test, genAccuracy, realAccuracy], feed_dict={all_real_data_conv: images, all_real_label_conv: labels, generated_labels_conv: genRandomLabels(BATCH_SIZE,CLASSES)})
            dev_disc_costs.append(_dev_disc_cost)
            lib.plot.plot('dev disc cost', np.mean(dev_disc_costs))
            lib.plot.plot('wgan dev disc cost', _dev_disc_cost_test)
            lib.plot.plot('dev class cost', _class_cost_test)
            lib.plot.plot('dev generated class cost', _gen_class_cost)
            lib.plot.plot('dev gen  cost', _dev_gen_cost_test)
            lib.plot.plot('dev gen accuracy', _dev_genAccuracy)
            lib.plot.plot('dev real accuracy', _dev_realAccuracy)


        if iteration % 1000 == 999:
            generate_image(iteration)
            #Can add generate_good_images method in here if desired

        if (iteration < 10) or (iteration % 100 == 99):
            lib.plot.flush()

        lib.plot.tick()

NOW MAKE A GAN FOR THE PYSEAS PROJECT IN PYTHON
This code is a partial implementation of a GAN (generative adversarial network) in TensorFlow. It contains the necessary components for building a GAN, such as the generator and discriminator networks, but it is missing some key parts of the training loop and other details.

To use this code as a basis for making your own GAN, you would need to add additional code to complete the implementation. This could include implementing the training loop and any necessary preprocessing or postprocessing steps, defining the loss function and optimization algorithm, and other details needed to train and evaluate the GAN. You would also need to customize the code to suit the specific requirements of your GAN, such as the type of GAN (ACGAN or WGAN), the dataset you are using, and the model architecture.




Graham Waters
How do I start doing that?
To complete the implementation of the GAN and use it for training, you can start by adding the following components to the code:

The training loop: This is the core part of the code where the GAN is trained. The training loop typically involves iterating over the training dataset, feeding the data into the GAN, updating the model weights and biases based on the loss function and optimization algorithm, and performing other necessary steps (such as saving checkpoints or evaluating the model) at regular intervals.

The loss function and optimization algorithm: The GAN uses a loss function to measure the difference between the generated and true images, and an optimization algorithm (such as stochastic gradient descent or Adam) to adjust the model weights and biases in order to minimize the loss. You can define these components in the code, either by using existing TensorFlow functions or by implementing them yourself.

Preprocessing and postprocessing steps: You may need to add preprocessing and postprocessing steps to the code to prepare the data for input to the GAN and postprocess the generated images for evaluation or visualization. These steps can include things like loading the images from a file, resizing or cropping them, converting them to a specific format, and applying any necessary transformations to extract features or improve the quality of the data.

Customization and fine-tuning: To make the GAN work well for your specific use case, you may need to customize and fine-tune the code to suit the requirements of your dataset and the desired properties of the generated images. This can involve changing the model architecture, adjusting the hyperparameters of the GAN, and experimenting with different loss functions and optimization algorithms to find the best combination for your specific problem.

Overall, completing the implementation of the GAN and using it for training involves a combination of adding the necessary components, customizing and fine-tuning the code, and experimenting with different settings to find the best configuration for your problem. It's often helpful to consult existing resources (such as tutorials or research papers) and use existing libraries or tools (such as TensorFlow or PyTorch) to make the process of building and training a GAN easier.




Graham Waters
How could I use tensorflow to make a GAN for this?
To use TensorFlow to make a GAN, you can follow these steps:

Define the model architecture: The first step is to define the architecture of the GAN, including the generator and discriminator networks. This typically involves deciding on the number and size of the layers in each network, as well as the specific types of layers (such as convolutional or fully-connected layers) and the activation functions that will be used.

Prepare the data: Before you can train the GAN, you need to gather a dataset of images that you want to use for training. This typically involves downloading and organizing the images into a structured format, such as folders for each class of images. You may also need to preprocess the images to resize them or perform other transformations to make them suitable for input to the GAN.

Define the loss function and optimization algorithm: Next, you need to define the loss function and optimization algorithm that will be used to train the GAN. The loss function measures the difference between the generated and true images, and the optimization algorithm adjusts the model weights and biases in order to minimize the loss. You can use existing TensorFlow functions or implement these components yourself.

Build the TensorFlow computation graph: Once the model architecture, data, and training components are defined, you can use TensorFlow to build the computation graph for the GAN. This involves specifying the inputs, outputs, and operations of the GAN in a TensorFlow graph, and defining any necessary placeholders, variables, or other components.

Train the GAN: After the computation graph is built, you can use TensorFlow to train the GAN. This typically involves iterating over the training data, feeding the data into the GAN, and using the optimization algorithm to update the model weights and biases based on the loss function. You can also use TensorFlow to evaluate the performance of the GAN on a separate dataset, and adjust the model or training settings as needed to improve its performance.

Overall, using TensorFlow to make a GAN involves defining the model architecture, preparing the data, defining the loss function and optimization algorithm, building the computation graph, and training the GAN. TensorFlow provides a range of tools and libraries that can make it easier to build and train a GAN, and you can use these resources to develop a GAN that works well for your specific use case.




Graham Waters
can you show me how step one looks in python3?
Here is an example of how step one (defining the model architecture) might look in Python 3 for a GAN:

Copy code
# Import necessary libraries
import tensorflow as tf

# Define the generator and discriminator networks
def generator(z):
  # Define the generator network architecture here
  # z is the input noise vector
  # The output of the generator is a generated image

def discriminator(x):
  # Define the discriminator network architecture here
  # x is the input image
  # The output of the discriminator is a score indicating the probability that the input image is real
This code defines the generator and discriminator functions, which represent the architecture of the GAN. The generator function takes a noise vector as input and generates a fake image as output, while the discriminator function takes an image as input and outputs a score indicating the probability that the image is real. You can customize the architecture of the networks by defining the specific types of layers, activation functions, and other details needed to build the GAN.




Graham Waters
Okay, what about step 2?
Here is an example of how step two (preparing the data) might look in Python 3 for a GAN:

Copy code
# Import necessary libraries
import tensorflow as tf
import numpy as np

# Load the dataset of images
def load_data(data_dir):
  # Use TensorFlow or other libraries to load the images from data_dir
  # Preprocess the images as needed (resize, convert to grayscale, etc.)
  # Return the preprocessed images as a NumPy array

# Split the dataset into training and evaluation sets
def split_dataset(images):
  # Use TensorFlow or other libraries to split the images into training and evaluation sets
  # Return the training and evaluation sets as separate NumPy arrays
This code defines two functions, load_data and split_dataset, which can be used to prepare the dataset of images for use in training the GAN. The load_data function loads the images from a specified directory, performs any necessary preprocessing steps, and returns the preprocessed images as a NumPy array. The split_dataset function then splits the images into training and evaluation sets, which can be used for training and evaluating the GAN respectively. You can customize these functions as needed to suit the specific requirements of your dataset and the GAN.




Graham Waters
And what about step 3?
Here is an example of how step three (defining the loss function and optimization algorithm) might look in Python 3 for a GAN:

Copy code
# Import necessary libraries
import tensorflow as tf

# Define the loss function
def gan_loss(logits_real, logits_fake):
  # Use TensorFlow or other libraries to define the loss function for the GAN
  # The loss function should measure the difference between the generated and true images
  # Return the loss as a TensorFlow tensor

# Define the optimization algorithm
def gan_optimizer(loss, learning_rate):
  # Use TensorFlow or other libraries to define the optimization algorithm for the GAN
  # The optimization algorithm should adjust the model weights and biases in order to minimize the loss
  # Return the optimizer as a TensorFlow optimizer
This code defines the gan_loss and gan_optimizer functions, which represent the loss function and optimization algorithm for the GAN. The gan_loss function defines the loss function for the GAN, which measures the difference between the generated and true images. The gan_optimizer function defines the optimization algorithm, which adjusts the model weights and biases in order to minimize the loss. You can customize these functions as needed to suit the specific requirements of your GAN, such as the type of GAN (ACGAN or WGAN) and the desired properties of the generated images.




Graham Waters
and step 4?
Here is an example of how step four (building the TensorFlow computation graph) might look in Python 3 for a GAN:

Copy code
# Import necessary libraries
import tensorflow as tf

# Define the input and output placeholders for the GAN
z = tf.placeholder(tf.float32, shape=[None, z_dim]) # Noise vector
x = tf.placeholder(tf.float32, shape=[None, img_size, img_size, channels]) # Real images

# Define the generator and discriminator networks
gen_images = generator(z)
logits_real, logits_fake = discriminator(x), discriminator(gen_images)

# Define the loss function and optimization algorithm
loss = gan_loss(logits_real, logits_fake)
optimizer = gan_optimizer(loss, learning_rate)

# Define the TensorFlow session and initialize the variables
sess = tf.Session()
sess.run(tf.global_variables_initializer())
This code builds the TensorFlow computation graph for the GAN. It defines the input and output placeholders for the GAN, and uses the generator and discriminator networks to compute the generated and true images. It also defines the loss function and optimization algorithm for the GAN, and creates a TensorFlow session to run the computation graph. You can customize this code as needed to suit the specific requirements of your GAN, such as the model architecture, data, and training settings.




Graham Waters
And finally, what about step 5?
Here is an example of how step five (training the GAN) might look in Python 3 for a GAN:

Copy code
# Import necessary libraries
import tensorflow as tf

# Load the dataset of images and split it into training and evaluation sets
images = load_data(data_dir)
train_images, eval_images = split_dataset(images)

# Train the GAN for a specified number of iterations
for i in range(num_iterations):
  # Sample a batch of noise vectors and real images from the dataset
  z_batch = sample_noise(batch_size, z_dim)
  x_batch = sample_images(train_images, batch_size)

  # Run the optimizer to update the model weights and biases based on the loss function
  _, loss_val = sess.run([optimizer, loss], feed_dict={z: z_batch, x: x_batch})

  # Print the loss and generate some sample images at regular intervals
  if i % print_interval == 0:
    print('Iteration %d: loss = %f' % (i, loss_val))
    sample_images = sess.run(gen_images, feed_dict={z: sample_noise(batch_size, z_dim)})
    save_images(sample_images, 'sample_%d.png' % i)

# Evaluate the GAN on the evaluation set
eval_loss = sess.run(loss, feed_dict={z: sample_noise(batch_size, z_dim), x: eval_images})
print('Evaluation loss: %f' % eval_loss)
This code trains the GAN for a specified number of iterations. It loads the dataset of images and splits it into




Graham Waters
Excellent! I am working on the generator function now.
how would you write it?
Here is an example of how the generator function might look in Python 3 for a GAN:

Copy code
# Import necessary libraries
import tensorflow as tf

def generator(z):
  # Define the generator network architecture here
  with tf.variable_scope('generator'):
    # Use fully-connected layers to transform the noise vector into a latent space
    h1 = tf.layers.dense(z, units=4*4*256, activation=tf.nn.relu)
    h1 = tf.reshape(h1, shape=[-1, 4, 4, 256])

    # Use transposed convolutional layers to upsample the image
    h2 = tf.layers.conv2d_transpose(h1, filters=128, kernel_size=5, strides=2, padding='same', activation=tf.nn.relu)
    h3 = tf.layers.conv2d_transpose(h2, filters=64, kernel_size=5, strides=2, padding='same', activation=tf.nn.relu)
    h4 = tf.layers.conv2d_transpose(h3, filters=32, kernel_size=5, strides=2, padding='same', activation=tf.nn.relu)
    h5 = tf.layers.conv2d_transpose(h4, filters=16, kernel_size=5, strides=2, padding='same', activation=tf.nn.relu)

    # Use a convolutional layer to produce the final output image
    gen_images = tf.layers.conv2d_transpose(h5, filters=3, kernel_size=5, strides=1, padding='same', activation=tf.tanh)

  return gen_images
This code defines the generator function, which represents the architecture of the generator network in the GAN. It uses fully-connected and transposed convolutional layers to transform the input noise vector into a generated image. The generated image is produced by upsampling the noise vector using transposed convolutional layers, and then applying a final convolutional layer to produce the final output image. You can customize this code as needed to suit the specific requirements of your GAN, such as the size and resolution of the generated images, and the specific types of layers and activation functions used.




Graham Waters
What about the discriminator function?
Here is an example of how the discriminator function might look in Python 3 for a GAN:

Copy code
# Import necessary libraries
import tensorflow as tf

def discriminator(x):
  # Define the discriminator network architecture here
  with tf.variable_scope('discriminator'):
    # Use convolutional layers to extract features from the input image
    h1 = tf.layers.conv2d(x, filters=16, kernel_size=5, strides=2, padding='same', activation=tf.nn.leaky_relu)
    h2 = tf.layers.conv2d(h1, filters=32, kernel_size=5, strides=2, padding='same', activation=tf.nn.leaky_relu)
    h3 = tf.layers.conv2d(h2, filters=64, kernel_size=5, strides=2, padding='same', activation=tf.nn.leaky_relu)
    h4 = tf.layers.conv2d(h3, filters=128, kernel_size=5, strides=2, padding='same', activation=tf.nn.leaky_relu)
    h5 = tf.layers.conv2d(h4, filters=256, kernel_size=5, strides=2, padding='same', activation=tf.nn.leaky_relu)

    # Use a fully-connected layer to produce the output logits
    logits = tf.layers.dense(h5, units=1, activation=None)

  return logits
This code defines the discriminator function, which represents the architecture of the discriminator network in the GAN. It uses convolutional layers to extract features from the input image, and then applies a fully-connected layer to produce the output logits. The logits represent the probability that the input image is real, and are used by the GAN to calculate the loss and optimize the model. You can customize this code as needed to suit the specific requirements of your GAN, such as the specific types of layers and activation functions used.




Graham Waters
okay I defined those functions.
What do you recommend for this function:
def load_data(data_dir):
    # Use TensorFlow or other libraries to load the images from data_dir
    # Preprocess the images as needed (resize, convert to grayscale, etc.)
    # Return the preprocessed images as a NumPy array
    pass
Here is an example of how the load_data function might look in Python 3 for a GAN:

Copy code
# Import necessary libraries
import tensorflow as tf
import numpy as np

def load_data(data_dir):
  # Load the images from the specified data directory
  filenames = tf.gfile.Glob(data_dir + '/*.jpg')
  images = [tf.gfile.FastGFile(filename, 'rb').read() for filename in filenames]

  # Decode the images and convert them to grayscale
  images = [tf.image.decode_jpeg(image, channels=1) for image in images]

  # Resize the images to a common size
  images = [tf.image.resize_images(image, [img_size, img_size]) for image in images]

  # Convert the images to a NumPy array
  images = np.array(images)

  return images
This code defines the load_data function, which loads the images from the specified data directory and preprocesses them as needed. It uses TensorFlow to load the images from the data directory, decode them, convert them to grayscale, resize them, and convert them to a NumPy array. You can customize this code as needed to suit the specific requirements of your GAN, such as the format and location of the images, the preprocessing steps applied to the images, and the desired size and resolution of the images.




Graham Waters
what about this one?
# Split the dataset into training and evaluation sets
def split_dataset(images):
    # Use TensorFlow or other libraries to split the images into training and evaluation sets
    # Return the training and evaluation sets as separate NumPy arrays
    pass

Please add your explanations to the docstring of the function and make the ratio a variable that is passed.
Here is an example of how the split_dataset function might look in Python 3 for a GAN:

Copy code
# Import necessary libraries
import tensorflow as tf
import numpy as np

def split_dataset(images, ratio):
  """
  Split the images into training and evaluation sets.
  Args:
    images: A NumPy array of images to split.
    ratio: A float value representing the ratio of the evaluation set size to the total dataset size.
  Returns:
    train_images: A NumPy array of training images.
    eval_images: A NumPy array of evaluation images.
  """

  # Shuffle the images and split them into training and evaluation sets
  indices = np.random.permutation(len(images))
  split_index = int(len(images) * (1 - ratio))
  train_images = images[indices[:split_index]]
  eval_images = images[indices[split_index:]]

  return train_images, eval_images
This code defines the split_dataset function, which splits a dataset of images into training and evaluation sets. It shuffles the images and uses a specified ratio to determine the split point between the training and evaluation sets. The function returns the training and evaluation sets as separate NumPy arrays. You can customize this code as needed to suit the specific requirements of your GAN, such as the desired split ratio, the types of images in the dataset, and the desired size and resolution of the images.




Graham Waters
# Define the loss function
def gan_loss(logits_real, logits_fake):
    # Use TensorFlow or other libraries to define the loss function for the GAN
    # The loss function should measure the difference between the generated and true images
    # Return the loss as a TensorFlow tensor
    pass


Please add your explanations to the docstring of the function and make the ratio a variable that is passed.
Here is an example of how the gan_loss function might look in Python 3 for a GAN:

Copy code
# Import necessary libraries
import tensorflow as tf

def gan_loss(logits_real, logits_fake):
  """
  Define the GAN loss function.
  Args:
    logits_real: A TensorFlow tensor representing the logits for the real images.
    logits_fake: A TensorFlow tensor representing the logits for the generated images.
  Returns:
    loss: A TensorFlow tensor representing the loss for the GAN.
  """

  # Compute the cross-entropy loss for the real and fake images
  loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_real, labels=tf.ones_like(logits_real)))
  loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_fake, labels=tf.zeros_like(logits_fake)))

  # Compute the total GAN loss
  loss = loss_real + loss_fake

  return loss
This code defines the gan_loss function, which defines the loss function for the GAN. It computes the cross-entropy loss for the real and generated images, and returns the total GAN loss as a TensorFlow tensor. The loss function measures the difference between the generated and true images, and is used to optimize the GAN model during training. You can customize this code as needed to suit the specific requirements of your GAN, such as the type of loss function used, and the specific parameters and settings for the loss function.




Graham Waters
# Define the optimization algorithm
def gan_optimizer(loss, learning_rate):
    # Use TensorFlow or other libraries to define the optimization algorithm for the GAN
    # The optimization algorithm should adjust the model weights and biases in order to minimize the loss
    # Return the optimizer as a TensorFlow optimizer
    pass

Please add your explanations to the docstring of the function comment the code with PEP8
Here is an example of how the gan_optimizer function might look in Python 3 for a GAN:

Copy code
# Import necessary libraries
import tensorflow as tf

def gan_optimizer(loss, learning_rate):
  """
  Define the GAN optimization algorithm.
  Args:
    loss: A TensorFlow tensor representing the GAN loss.
    learning_rate: A float value representing the learning rate for the optimizer.
  Returns:
    optimizer: A TensorFlow optimizer for the GAN.
  """

  # Define the optimization algorithm
  optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

  # Compute the gradients
  grads_and_vars = optimizer.compute_gradients(loss)

  # Apply the gradients to adjust the model weights and biases
  train_op = optimizer.apply_gradients(grads_and_vars)

  return train_op
This code defines the gan_optimizer function, which defines the optimization algorithm for the GAN. It uses the Adam optimizer to compute the gradients for the model weights and biases, and applies the gradients to adjust the model in order to minimize the loss. The function returns the optimizer as a TensorFlow optimizer. You can customize this code as needed to suit the specific requirements of your GAN, such as the type of optimizer used, and the specific parameters and settings for the optimizer.



🤔
How do I apply style transfer from the painter Titian?
I have paintings from Titian in a folder`../images/titian/` and I want to create new ocean images that are made with a GAN trained on the images in `../images/ocean/`?

Save new edit
Cancel



Graham Waters

2 / 2

I realize that this GAN is not specifically designed for style transfer. This GAN is for image generation. So, the following question applies to step 2 of this project, where we take the image that our GAN generated and apply a style to it. What is the easiest method of applying style transfer to a generated image?

Assets
I have paintings from Titian in a folder`../images/titian/`
The new (fake) ocean images made with the model we created above (a GAN which was trained on the images in `../images/ocean/`)
The GAN model.
​



···
Free Research Preview: ChatGPT is optimized for dialogue. Our goal is to make AI systems more natural to interact with, and your feedback will help us improve our systems and make them safer.
Reset Thread
Light Mode
OpenAI Discord
Updates & FAQ
Log out
