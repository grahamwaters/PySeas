# image_classifier.py
import PIL
import numpy as np
from tensorflow.keras.preprocessing import image as keras_image
import tensorflow as tf
from tensorflow.keras.models import load_model

IMAGE_SIZE = (224, 224)
CLASS_NAMES = [
    "sunset",
    "night",
    "white",
    "storm",
    "clouds",
    "normal",
    "moon"
]
temp_model = load_model('models/blank_or_not_model/keras_model.h5')

class ImageClassifier:
    def __init__(self):
        pass

    def classify_image(self, image, model, white_model=temp_model, image_size=IMAGE_SIZE, class_names=CLASS_NAMES):
        if not isinstance(image, PIL.Image.Image):
            img = keras_image.load_img(image, target_size=image_size)
        else:
            img = image.resize(image_size)

        if img.mode != 'RGB':
            img = img.convert('RGB')

        img_array = keras_image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        avg_color = np.array(img).mean(axis=(0, 1))
        if avg_color.mean() > 350:
            return 'white'
        elif avg_color.mean() < 10:
            return 'night'
        else:
            # blank_or_not_probabilities = white_model.predict(img_array)
            # for i in range(0, 6):
            #     if blank_or_not_probabilities[0][i] > 0.5:
            #         return 'White'

            # blank_or_not_class_index = np.argmax(blank_or_not_probabilities)
            # if blank_or_not_class_index == 0:
            #     return 'White'

            probabilities = model.predict(img_array)
            class_index = np.argmax(probabilities)
            return class_names[class_index]
