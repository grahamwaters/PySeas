import cv2

class ImageProcessor:
    """
     the ImageProcessor class includes three methods: resize_image, crop_image, and color_correct. These methods use the cv2 module to perform the specified image processing tasks, and return the processed images as output.

        The resize_image method takes an image and the desired width and height as inputs, and uses the cv2.resize function to resize the image to the specified dimensions. The crop_image method takes an image and the coordinates of the top-left and bottom-right corners of the region to be cropped, and uses array slicing to crop the image. The color_correct method takes an image as input and uses the cv2.cvtColor function to apply color correction to the image.

        By implementing these methods in your ImageProcessor class, you can create a useful tool for preprocessing and color correcting the images from the NOAA buoys. You can then use this class in your other code to improve the quality of the images and make them more suitable for training or generating panoramic images.

    """
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
