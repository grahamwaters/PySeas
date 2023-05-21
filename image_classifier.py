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
    "storm",
    "clouds",
    "clear",
    "object",
    "dusk",
    "beautiful"
]

# 0 Direct Sun
# 1 Stormy Weather
# 2 Interesting
# 3 Object Detected
# 4 Sunset
# 5 Clouds
# 6 Night
CLASS_NAMES = [
    "Direct Sun",
    "Stormy Weather",
    "Interesting",
    "Object Detected",
    "Sunset",
    "Clouds",
    "Night"
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
        # if avg_color.mean() > 350:
        #     return 'white'
        # elif avg_color.mean() < 10:
        #     return 'night'
        # else:
        probabilities = model.predict(img_array)
        class_index = np.argmax(probabilities)
        return class_names[class_index]
