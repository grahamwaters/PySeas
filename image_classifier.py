# image_classifier.py
# keras_image
# import numpy as np

import PIL
import numpy as np
from tensorflow.keras.preprocessing import image as keras_image
import tensorflow as tf
from tensorflow.keras.models import load_model

IMAGE_SIZE = (224, 224)
# CLASS_NAMES = ['Sunset', 'Storms', 'Normal', 'Object In View']
CLASS_NAMES = [
    "sunset",
    "night",
    "white",
    "storm",
    "clouds",
    "strange sky",
    "object",
    "normal"
]

class ImageClassifier:
    def __init__(self):
        # Initialize your image classifier here
        pass

    # Classification Functions
    def classify_image(self, image, model, blank_or_not_model, image_size=IMAGE_SIZE, class_names=CLASS_NAMES):
        """
        The classify_image function takes in an image and a model, and returns the class of the image.

        :param self: Represent the instance of the object itself
        :param image: Pass the image to be classified
        :param model: Pass in the model that will be used to classify the image
        :param blank_or_not_model: Check if the image is blank or not
        :param image_size: Resize the image to a standard size
        :param class_names: Return the name of the class that is predicted by our model
        :return: A string of the best guess
        :doc-author: Trelent
        """

        if not isinstance(image, PIL.Image.Image):
            img = keras_image.load_img(image, target_size=image_size)
        else:
            img = image.resize(image_size)

        # check if blank or not first with blank_or_not_model. If blank, return 'Blank'



        if img.mode != 'RGB':
            img = img.convert('RGB')

        img_array = keras_image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        # blank or not model
        blank_or_not_probabilities = blank_or_not_model.predict(img_array)
        blank_or_not_class_index = np.argmax(blank_or_not_probabilities)
        if blank_or_not_class_index == 0:
            #print(f"Blank image")
            return 'White'


        probabilities = model.predict(img_array)
        class_index = np.argmax(probabilities)
        print(f"Best guess: {class_names[class_index]}")
        return class_names[class_index]
