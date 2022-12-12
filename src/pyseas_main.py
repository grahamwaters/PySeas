from BuoyData import BuoyData
from ImageProcessor import ImageProcessor
from Classifier import Classifier
from PanoramaGenerator import PanoramaGenerator
import numpy as np







# create a BuoyData instance to manage the buoy data
buoy_data = BuoyData("44011", "Buoy Name", "37.75", "-122.37", "Buoy Data")

print(f'Step 1: Downloading data for buoy {buoy_data.buoy_id}...')
# download the data for a particular buoy using the NOAA API
buoy_data.download_data()

print(f'Step 2: Storing data for buoy {buoy_data.buoy_id}...')

# store the downloaded data in a local database
buoy_data.store_data()

# retrieve the data for the buoy from the local database
data = buoy_data.get_data()

print(f'Number of images: {len(data)}')
# create an ImageProcessor instance to pre-process the images
image_processor = ImageProcessor()

# pre-process the images in the buoy data
data = image_processor.pre_process(data)

# print(f'classes: {data.classes}'')
# create a Classifier instance to classify the images as sunsets or non-sunsets
classifier = Classifier()

print(f'training the ')
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
