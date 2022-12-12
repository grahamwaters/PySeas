# Project PySeas

# Project PySeas I want to make a stunning, evolving art composition out of the images generated every hour by the buoys in the ocean, which the National Oceanographic Atmospheric Association or NOAA maintains. They have an API that allows you to call that API and pull images from buoys worldwide. I want to be able to take those images every hour from the buoys and stitch them together using Open CV in a way that makes it I would like to be able to stitch them together to create a panoramic photo, but not by just concatenating the images because that doesn't always look good. Also, I would like to match the images' horizon lines because the buoy images are uneven due to the waves. It is usually rocking back and forth.

# To accomplish this, I plan to use Open CV's image stitching algorithms to blend the buoy images seamlessly. I will also use Open CV's horizon detection and correction algorithms to ensure that the final panoramic photo looks natural and aesthetically pleasing.
# To make the project even more interesting, I want to add a time-lapse element to it. This means that the panoramic image will evolve over time as the buoy images change. I plan to use Open CV's image blending algorithms to smoothly transition from one shot to the next, creating a dynamic and ever-changing art piece.
# I also want to add an interactive element to the project. I plan to create a website where users can explore the panoramic image and see different sections of it in detail. The website will also include information about the buoy locations and the weather conditions at those locations.
# Overall, my goal with Project PySeas is to create a unique and beautiful art piece that showcases the beauty of the ocean and the power of technology. I can create something extraordinary and captivating by combining the images from the buoys with advanced image processing algorithms.

# Import necessary libraries

import tensorflow as tf
# Define the generator and discriminator networks

def generator(z):
    """
    generator This code defines the generator function, which represents the architecture of the generator network in the GAN. It uses fully-connected and transposed convolutional layers to transform the input noise vector into a generated image. The generated image is produced by upsampling the noise vector using transposed convolutional layers, and then applying a final convolutional layer to produce the final output image. You can customize this code as needed to suit the specific requirements of your GAN, such as the size and resolution of the generated images, and the specific types of layers and activation functions used.

    Args:
        z: The input noise vector
    Returns:
        gen_images: The generated images
    """
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


def discriminator(x):
    """
    discriminator This code defines the discriminator function, which represents the architecture of the discriminator network in the GAN. It uses convolutional layers to extract features from the input image, and then applies a fully-connected layer to produce the output logits. The logits represent the probability that the input image is real, and are used by the GAN to calculate the loss and optimize the model. You can customize this code as needed to suit the specific requirements of your GAN, such as the specific types of layers and activation functions used.
    """
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


#! Step 2 - Gather
# This code defines two functions, load_data and split_dataset, which can be used to prepare the dataset of images for use in training the GAN. The load_data function loads the images from a specified directory, performs any necessary preprocessing steps, and returns the preprocessed images as a NumPy array. The split_dataset function then splits the images into training and evaluation sets, which can be used for training and evaluating the GAN respectively. You can customize these functions as needed to suit the specific requirements of your dataset and the GAN.
# Import necessary libraries

import numpy as np

# Load the dataset of images
# Import necessary libraries

def load_data(data_dir):
    """
    This code defines the load_data function, which loads the images from the specified data directory and preprocesses them as needed. It uses TensorFlow to load the images from the data directory, decode them, convert them to grayscale, resize them, and convert them to a NumPy array. You can customize this code as needed to suit the specific requirements of your GAN, such as the format and location of the images, the preprocessing steps applied to the images, and the desired size and resolution of the images.

    """
    # Load the images from the specified data directory
    filenames = tf.gfile.Glob(data_dir + '/*.jpg') #note: this does not find .png files yet.
    images = [tf.gfile.FastGFile(filename, 'rb').read() for filename in filenames]

    # Decode the images and convert them to grayscale
    images = [tf.image.decode_jpeg(image, channels=1) for image in images]

    # Resize the images to a common size
    images = [tf.image.resize_images(image, [img_size, img_size]) for image in images]

    # Convert the images to a NumPy array
    images = np.array(images)

    return images

def split_dataset(images):
    # Shuffle the images and split them into training and evaluation sets
    np.random.shuffle(images)
    train_images = images[:int(0.8*len(images))]
    eval_images = images[int(0.8*len(images)):]

    return train_images, eval_images

#* option 2 for this function
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

#! Step 3
# This code defines the gan_loss and gan_optimizer functions, which represent the loss function and optimization algorithm for the GAN. The gan_loss function defines the loss function for the GAN, which measures the difference between the generated and true images. The gan_optimizer function defines the optimization algorithm, which adjusts the model weights and biases in order to minimize the loss. You can customize these functions as needed to suit the specific requirements of your GAN, such as the type of GAN (ACGAN or WGAN) and the desired properties of the generated images.
# Import necessary libraries


def gan_loss(logits_real, logits_fake):
  """
  Define the GAN loss function. This code defines the gan_loss function, which defines the loss function for the GAN. It computes the cross-entropy loss for the real and generated images, and returns the total GAN loss as a TensorFlow tensor. The loss function measures the difference between the generated and true images, and is used to optimize the GAN model during training. You can customize this code as needed to suit the specific requirements of your GAN, such as the type of loss function used, and the specific parameters and settings for the loss function.
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



def gan_optimizer(loss, learning_rate):
    """
    Define the GAN optimization algorithm. This code defines the gan_optimizer function, which defines the optimization algorithm for the GAN. It uses the Adam optimizer to compute the gradients for the model weights and biases, and applies the gradients to adjust the model in order to minimize the loss. The function returns the optimizer as a TensorFlow optimizer. You can customize this code as needed to suit the specific requirements of your GAN, such as the type of optimizer used, and the specific parameters and settings for the optimizer.
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


#! Step 4 - Computation Graph
# This code builds the TensorFlow computation graph for the GAN. It defines the input and output placeholders for the GAN, and uses the generator and discriminator networks to compute the generated and true images. It also defines the loss function and optimization algorithm for the GAN, and creates a TensorFlow session to run the computation graph. You can customize this code as needed to suit the specific requirements of your GAN, such as the model architecture, data, and training settings.

# Import necessary libraries


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

#! Step 5 - Training the GAN
# Import necessary libraries


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


#! Step 6 - Generating Images
# This code defines the generator function, which generates images using the trained generator network. The function takes as input a set of noise vectors, and returns a set of generated images. You can customize this code as needed to suit the specific requirements of your GAN, such as the model architecture, data, and training settings.


#! Step 7 - Style Transfer from the Titian Paintings Folder
# This code defines the style_transfer function, which performs style transfer on a set of images. The function takes as input a set of images, and returns a set of images that have been transformed to match the style of the images in the style_images folder. You can customize this code as needed to suit the specific requirements of your style transfer model, such as the model architecture, data, and training settings.

# Import necessary libraries
import tensorflow as tf

# Load the pre-trained style transfer model
model = tf.keras.models.load_model('style_transfer_model.h5')

# Load the Titian images
titian_images = tf.keras.preprocessing.image.ImageDataGenerator().flow_from_directory('../images/titian/')

# Preprocess the images
titian_images = tf.image.resize(titian_images, (256, 256))

"""
You can then use the pre-trained style transfer model to apply the style of the Titian paintings to the generated ocean images. This is typically done by feeding the style and content images to the model, and then running the model to generate the stylized output image. Here is an example of how this might look in Python 3 using TensorFlow.
"""

titian_images = tf.keras.applications.mobilenet_v2.preprocess_input(titian_images)

# Load the generated ocean images from the BuoyGAN
ocean_images = tf.keras.preprocessing.image.ImageDataGenerator().flow_from_directory('../images/ocean/')

# Preprocess the images
ocean_images = tf.image.resize(ocean_images, (256, 256))

# Use the model to apply the style of the Titian paintings to the generated ocean images
stylized_images = model.predict([titian_images, ocean_images])

# Save the stylized images
tf.keras.preprocessing.image.save_img('stylized_image.jpg', stylized_images)
"""
This code uses the pre-trained style transfer model to apply the style of the Titian paintings to the generated ocean images. It then saves the stylized images as new image files that can be used as needed. You can customize this code as needed to suit the specific requirements of your project. For example, you could adjust the model settings, save the images to a different location, or apply the style transfer to a different set of images.
"""
