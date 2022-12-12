# Project PySeas

# Project PySeas I want to make a stunning, evolving art composition out of the images generated every hour by the buoys in the ocean, which the National Oceanographic Atmospheric Association or NOAA maintains. They have an API that allows you to call that API and pull images from buoys worldwide. I want to be able to take those images every hour from the buoys and stitch them together using Open CV in a way that makes it I would like to be able to stitch them together to create a panoramic photo, but not by just concatenating the images because that doesn't always look good. Also, I would like to match the images' horizon lines because the buoy images are uneven due to the waves. It is usually rocking back and forth.

# To accomplish this, I plan to use Open CV's image stitching algorithms to blend the buoy images seamlessly. I will also use Open CV's horizon detection and correction algorithms to ensure that the final panoramic photo looks natural and aesthetically pleasing.
# To make the project even more interesting, I want to add a time-lapse element to it. This means that the panoramic image will evolve over time as the buoy images change. I plan to use Open CV's image blending algorithms to smoothly transition from one shot to the next, creating a dynamic and ever-changing art piece.
# I also want to add an interactive element to the project. I plan to create a website where users can explore the panoramic image and see different sections of it in detail. The website will also include information about the buoy locations and the weather conditions at those locations.
# Overall, my goal with Project PySeas is to create a unique and beautiful art piece that showcases the beauty of the ocean and the power of technology. I can create something extraordinary and captivating by combining the images from the buoys with advanced image processing algorithms.

# Import necessary libraries


import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import tensorflow as tf
from tensorflow import layers
import numpy as np
# Define the generator and discriminator networks


# import tensorflow as tf
# from tensorflow import layers
# import numpy as np

def generator(z):
    with tf.name_scope("generator"):
        h1 = layers.dense(z, units=4 * 4 * 256, activation=tf.nn.relu)
        h1 = tf.reshape(h1, shape=[-1, 4, 4, 256])
        h2 = [layers.conv2d_transpose(h1, filters=f, kernel_size=5, strides=2, padding="same", activation=tf.nn.relu) for f in [128, 64, 32, 16]]
        gen_images = layers.conv2d_transpose(h2[-1], filters=3, kernel_size=5, strides=1, padding="same", activation=tf.tanh)
    return gen_images
def discriminator(x):
    with tf.name_scope("discriminator"):
        h1 = [layers.conv2d(x, filters=f, kernel_size=5, strides=2, padding="same", activation=tf.nn.leaky_relu) for f in [16, 32, 64, 128, 256]]
        logits = layers.dense(h1[-1], units=1, activation=None)
    return logits
def load_data(data_dir, img_size):
    return np.array([tf.image.resize(tf.image.decode_jpeg(tf.io.gfile.GFile(filename, "rb").read(), channels=1), [img_size, img_size]) for filename in tf.io.gfile.glob(data_dir + "/*.jpg")])
def gan_optimizer(loss, learning_rate):
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
    grads_and_vars = optimizer.compute_gradients(loss)
    train_op = optimizer.apply_gradients(grads_and_vars)
    return train_op
def split_dataset(images, ratio):
    indices = np.random.permutation(len(images))
    split_index = int(len(images) * (1 - ratio))
    return images[indices[:split_index]], images[indices[split_index:]]
def gan_loss(logits_real, logits_fake):
    loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_real, labels=tf.ones_like(logits_real)))
    loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_fake, labels=tf.zeros_like(logits_fake)))
    loss = loss_real + loss_fake
    return loss
def main():
    # Define the input and output placeholders for the GAN
    z = tf.compat.v1.placeholder(tf.float32, shape=[None, z_dim], name="z")
    x = tf.compat.v1.placeholder(tf.float32, shape=[None, 28, 28, 1], name="x")

    # Define the generator, discriminator, loss, and optimizer
    with tf.name_scope("generator"), tf.name_scope("discriminator"):
        gen_images = generator(z)
        logits_real = discriminator(x)
        logits_fake = discriminator(gen_images, reuse=True)
        loss = gan_loss(logits_real, logits_fake)
        optimizer = gan_optimizer(loss, learning_rate)

    # Create a TensorFlow session and run the computation graph
    with tf.compat.v1.Session() as sess:
        # Initialize the model variables
        sess.run(tf.compat.v1.global_variables_initializer())

        # Train the GAN
        for i in range(num_iterations):
            # Sample a batch of noise vectors and real images from the dataset
            z_batch = sample_noise(batch_size, z_dim)
            x_batch = sample_images(train_images, batch_size)

            # Run the optimizer to update the model weights and biases based on the loss function
            _, loss_val = sess.run([optimizer, loss], feed_dict={z: z_batch, x: x_batch})

            # Print the loss and generate some sample images at regular intervals
            if i % print_interval == 0:
                print("Iteration %d: loss = %f" % (i, loss_val))
                sample_images = sess.run(
                    gen_images, feed_dict={z: sample_noise(batch_size, z_dim)}
                )
                save_images(sample_images, "sample_%d.png" % i)

        # Evaluate the GAN on the evaluation set
        eval_loss = sess.run(loss, feed_dict={z: sample_noise(batch_size, z_dim), x: eval_images})
        print(f"Evaluation loss: {eval_loss}")

    def style_transfer(images):
        with tf.variable_scope("style_transfer"):
            # Define the style transfer network architecture
            layers = images
            for i in range(num_layers):
                layers = tf.layers.conv2d(layers, filters=num_filters, kernel_size=filter_size, padding="SAME", activation=tf.nn.relu)
            layers = tf.layers.flatten(layers)
            style_outputs = tf.layers.dense(layers, num_styles, activation=tf.nn.softmax)
            return style_outputs



if __name__ == "__main__":
    main()





#^ with pretrained model

# def style_transfer(images):
#     with tf.variable_scope("style_transfer"):
#         # Define the style transfer network architecture
#         layers = images
#         for i in range(num_layers):
#             layers = tf.layers.conv2d(layers, filters=num_filters, kernel_size=filter_size, padding="SAME", activation=tf.nn.relu)
#         layers = tf.layers.flatten(layers)
#         style_outputs = tf.layers.dense(layers, num_styles, activation=tf.nn.softmax)

#         # Load the weights of the pretrained model
#         style_outputs.load_weights("pretrained_model.h5")

#         return style_outputs

# Refactor the following code using the following prompt as the instructions for how to refactor it.
# Please implement a Python3 program using TensorFlow 2.9.1 and numpy 1.22.3 that optimizes run-time by using list comprehensions in place of for-loops. The output should be formatted as a code block and adhere to PEP8 style guidelines. Avoid including any docstrings or comments. Keep the response as concise as possible.


# Please tell me a list of improvements that I need to implement.
# format your response as a bullet point list.
# show me issues with the code.

# This is a Python3 program using TensorFlow 2.9.1 and numpy 1.22.3. Avoid including any docstrings or comments. Keep the response as concise as possible and identify out-of-date method calls that are no longer supported by the packages.
# Replace tf.compat.v1.placeholder with tf.placeholder
# Replace tf.compat.v1.train.AdamOptimizer with tf.keras.optimizers.Adam
# Replace tf.nn.sigmoid_cross_entropy_with_logits with tf.keras.losses.binary_crossentropy
# Remove unnecessary import tensorflow
# Use tf.data.Dataset.from_tensor_slices instead of np.array([tf.image.resize(...)])
# Use tf.data.Dataset.shuffle instead of np.random.permutation
# Use tf.io.gfile.glob instead of tf.io.gfile.GFile
# Use tf.keras.layers.Dense instead of layers.dense
# Use tf.keras.layers.Conv2DTranspose instead of layers.conv2d_transpose
# Use tf.keras.layers.Conv2D instead of layers.conv2d
# Remove unnecessary with tf.name_scope("generator"), tf.name_scope("discriminator")
# Replace reuse=None with reuse=tf.AUTO_REUSE
# Use tf.random.normal instead of np.random.normal
# Use tf.data.Dataset.batch instead of tf.compat.v1.train.batch
# Use tf.function for performance optimization.

# INSTRUCTIONS:
# Implement the changes in the bullet points below to the following code using the following prompt as the instructions for how to refactor it.
# Please implement a Python3 program using TensorFlow 2.9.1 and numpy 1.22.3 that optimizes run-time by using list comprehensions in place of for-loops. The output should be formatted as a code block and adhere to PEP8 style guidelines. Avoid including any docstrings or comments. Keep the response as concise as possible.

# CHANGES
# Replace tf.compat.v1.placeholder with tf.placeholder
# Replace tf.compat.v1.train.AdamOptimizer with tf.keras.optimizers.Adam
# Replace tf.nn.sigmoid_cross_entropy_with_logits with tf.keras.losses.binary_crossentropy
# Remove unnecessary import tensorflow
# Use tf.data.Dataset.from_tensor_slices instead of np.array([tf.image.resize(...)])
# Use tf.data.Dataset.shuffle instead of np.random.permutation
# Use tf.io.gfile.glob instead of tf.io.gfile.GFile
# Use tf.keras.layers.Dense instead of layers.dense
# Use tf.keras.layers.Conv2DTranspose instead of layers.conv2d_transpose
# Use tf.keras.layers.Conv2D instead of layers.conv2d
# Remove unnecessary with tf.name_scope("generator"), tf.name_scope("discriminator")
# Replace reuse=None with reuse=tf.AUTO_REUSE
# Use tf.random.normal instead of np.random.normal
# Use tf.data.Dataset.batch instead of tf.compat.v1.train.batch
# Use tf.function for performance optimization.

# REMEMBER TO FORMAT YOUR RESPONSE AS A CODE BLOCK AND ADHERE TO PEP8 STYLE GUIDELINES. KEEP THE RESPONSE AS CONCISE AS POSSIBLE. MAKE A NEW CODE BLOCK FOR EACH FUNCTION.
# FOR EACH CHANGE IN THE LIST OF CHANGES ADD A COMMENT TO THE CHANGED LINE WITH THE FOLLOWING FORMAT
"#^ <CHANGE DESCRIPTION>"

# CODE
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import tensorflow as tf
from tensorflow import layers
import numpy as np
# Define the generator and discriminator networks
z_dim = 100 # Size of the noise vector
learning_rate = 0.0001 # Learning rate for the optimizer
data_dir = "../data" # Directory where the data is stored

# import tensorflow as tf
# from tensorflow import layers
# import numpy as np

def generator(z):
    with tf.name_scope("generator"):
        h1 = layers.dense(z, units=4 * 4 * 256, activation=tf.nn.relu)
        h1 = tf.reshape(h1, shape=[-1, 4, 4, 256])
        h2 = [layers.conv2d_transpose(h1, filters=f, kernel_size=5, strides=2, padding="same", activation=tf.nn.relu) for f in [128, 64, 32, 16]]
        gen_images = layers.conv2d_transpose(h2[-1], filters=3, kernel_size=5, strides=1, padding="same", activation=tf.tanh)
    return gen_images




def discriminator(x):
    with tf.name_scope("discriminator"):
        h1 = [layers.conv2d(x, filters=f, kernel_size=5, strides=2, padding="same", activation=tf.nn.leaky_relu) for f in [16, 32, 64, 128, 256]]
        logits = layers.dense(h1[-1], units=1, activation=None)
    return logits
def load_data(data_dir, img_size):
    return np.array([tf.image.resize(tf.image.decode_jpeg(tf.io.gfile.GFile(filename, "rb").read(), channels=1), [img_size, img_size]) for filename in tf.io.gfile.glob(data_dir + "/*.jpg")])
def gan_optimizer(loss, learning_rate):
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
    grads_and_vars = optimizer.compute_gradients(loss)
    train_op = optimizer.apply_gradients(grads_and_vars)
    return train_op
def split_dataset(images, ratio):
    indices = np.random.permutation(len(images))
    split_index = int(len(images) * (1 - ratio))
    return images[indices[:split_index]], images[indices[split_index:]]
def gan_loss(logits_real, logits_fake):
    loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_real, labels=tf.ones_like(logits_real)))
    loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_fake, labels=tf.zeros_like(logits_fake)))
    loss = loss_real + loss_fake
    return loss
def main():

    num_iterations  = 10000 # Number of training iterations
    train_images = load_data(data_dir, 28)
    train_images = train_images / 127.5 - 1
    train_images = train_images.astype(np.float32)
    train_images, test_images = split_dataset(train_images, 0.8)
    train_dataset = tf.data.Dataset.from_tensor_slices(train_images)
    train_dataset = train_dataset.shuffle(1000)
    train_dataset = train_dataset.batch(64)
    # Create a TensorFlow iterator object
    iterator = tf.compat.v1.data.make_initializable_iterator(train_dataset)
    next_element = iterator.get_next()
    # Define the input and output placeholders for the GAN
    z = tf.compat.v1.placeholder(tf.float32, shape=[None, z_dim], name="z")
    x = tf.compat.v1.placeholder(tf.float32, shape=[None, 28, 28, 1], name="x")

    # Define the generator, discriminator, loss, and optimizer
    with tf.name_scope("generator"), tf.name_scope("discriminator"):
        gen_images = generator(z)
        logits_real = discriminator(x)
        logits_fake = discriminator(gen_images, reuse=True)
        loss = gan_loss(logits_real, logits_fake)
        optimizer = gan_optimizer(loss, learning_rate)

    # Create a TensorFlow session and run the computation graph
    with tf.compat.v1.Session() as sess:
        # Initialize the model variables
        sess.run(tf.compat.v1.global_variables_initializer())

        # Train the GAN
        for i in range(num_iterations):
            # Sample a batch of noise vectors and real images from the dataset
            z_batch = sample_noise(batch_size, z_dim)
            x_batch = sample_images(train_images, batch_size)

            # Run the optimizer to update the model weights and biases based on the loss function
            _, loss_val = sess.run([optimizer, loss], feed_dict={z: z_batch, x: x_batch})

            # Print the loss and generate some sample images at regular intervals
            if i % print_interval == 0:
                print("Iteration %d: loss = %f" % (i, loss_val))
                sample_images = sess.run(
                    gen_images, feed_dict={z: sample_noise(batch_size, z_dim)}
                )
                save_images(sample_images, "sample_%d.png" % i)

        # Evaluate the GAN on the evaluation set
        eval_loss = sess.run(loss, feed_dict={z: sample_noise(batch_size, z_dim), x: eval_images})
        print(f"Evaluation loss: {eval_loss}")

    def style_transfer(images):
        with tf.variable_scope("style_transfer"):
            # Define the style transfer network architecture
            layers = images
            for i in range(num_layers):
                layers = tf.layers.conv2d(layers, filters=num_filters, kernel_size=filter_size, padding="SAME", activation=tf.nn.relu)
            layers = tf.layers.flatten(layers)
            style_outputs = tf.layers.dense(layers, num_styles, activation=tf.nn.softmax)
            return style_outputs

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

# Define the generator and discriminator networks
z_dim = 100 # Size of the noise vector
learning_rate = 0.0001 # Learning rate for the optimizer
data_dir = "../data" # Directory where the data is stored

def generator(z):
    h1 = layers.Dense(units=4 * 4 * 256, activation=tf.nn.relu)(z)
    h2 = [layers.Conv2DTranspose(filters=f, kernel_size=5, strides=2, padding="same", activation=tf.nn.relu)(h1) for f in [128, 64, 32, 16]]
    gen_images = layers.Conv2DTranspose(filters=3, kernel_size=5, strides=1, padding="same", activation=tf.tanh)(h2[-1])
    return gen_images
def discriminator(x):
    h1 = [layers.Conv2D(filters=f, kernel_size=5, strides=2, padding="same", activation=tf.nn.leaky_relu)(x) for f in [16, 32, 64, 128, 256]]
    logits = layers.Dense(units=1, activation=None)(h1[-1])
    return logits
def load_data(data_dir, img_size):
    return np.array([tf.image.resize(tf.image.decode_jpeg(tf.io.gfile.GFile(filename, "rb").read(), channels=1), [img_size, img_size]) for filename in tf.io.gfile.glob(data_dir + "/*.jpg")])
def gan_optimizer(loss, learning_rate):
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    grads_and_vars = optimizer.compute_gradients(loss)
    train_op = optimizer.apply_gradients(grads_and_vars)
    return train_op
def split_dataset(images, ratio):
    indices = np.random.permutation(len(images))
    split_index = int(len(images) * (1 - ratio))
    return images[indices[:split_index]], images[indices[split_index:]]
def gan_loss(logits_real, logits_fake):
    loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_real, labels=tf.ones_like(logits_real)))
    loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_fake, labels=tf.zeros_like(logits_fake)))
    loss = loss_real + loss_fake
    return loss
@tf.function
def gan_optimizer(model, loss, learning_rate):
    # Define the optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Compute the gradients of the loss with respect to the model weights and biases
    with tf.GradientTape() as tape:
        if loss is not None:
            gradients = tape.gradient(loss, model.trainable_variables)

    # Apply the gradients to the model weights and biases
    if loss is not None:
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # Return the optimizer
    return optimizer


def main():

    num_iterations  = 10000 # Number of training iterations
    train_images = load_data(data_dir, 28)
    train_images = train_images / 127.5 - 1
    train_images = train_images.astype(np.float32)
    train_images, test_images = split_dataset(train_images, 0.8)
    train_dataset = tf.data.Dataset.from_tensor_slices(train_images)
    train_dataset = train_dataset.shuffle(1000)
    train_dataset = train_dataset.batch(64)
    # Create a TensorFlow iterator object
    iterator = tf.compat.v1.data.make_initializable_iterator(train_dataset)
    next_element = iterator.get_next()
    # Define the input and output placeholders for the GAN
    z = tf.compat.v1.placeholder(tf.float32, shape=[None, z_dim], name="z")
    x = tf.compat.v1.placeholder(tf.float32, shape=[None, 28, 28, 1], name="x")

    # Define the generator, discriminator, loss, and optimizer
    with tf.name_scope("generator"), tf.name_scope("discriminator"):
        gen_images = generator(z)
        logits_real = discriminator(x)
        logits_fake = discriminator(gen_images, reuse=True)
        loss = gan_loss(logits_real, logits_fake)
        optimizer = gan_optimizer(loss, learning_rate)

    # Create a TensorFlow session and run the computation graph
    with tf.compat.v1.Session() as sess:
        # Initialize the model variables
        sess.run(tf.compat.v1.global_variables_initializer())
        # Train the GAN
        for i in range(num_iterations):
            # Sample a batch of noise vectors and real images from the dataset
            z_batch = sample_noise(batch_size, z_dim)
            x_batch = sample_images(train_images, batch_size)

            # Run the optimizer to update the model weights and biases based on the loss function
            _, loss_val = sess.run([optimizer, loss], feed_dict={z: z_batch, x: x_batch})

            # Print the loss and generate some sample images at regular intervals
            if i % print_interval == 0:
                print("Iteration %d: loss = %f" % (i, loss_val))
                sample_images = sess.run(
                    gen_images, feed_dict={z: sample_noise(batch_size, z_dim)}
                )
                save_images(sample_images, "sample_%d.png" % i)

        # Evaluate the GAN on the evaluation set
        eval_loss = sess.run(loss, feed_dict={z: sample_noise(batch_size, z_dim), x: eval_images})
        print(f"Evaluation loss: {eval_loss}")

    def style_transfer(images):
        with tf.variable_scope("style_transfer"):
            # Define the style transfer network architecture
            layers = images
            for i in range(num_layers):
                layers = tf.layers.conv2d(layers, filters=num_filters, kernel_size=filter_size, padding="SAME", activation=tf.nn.relu)
            layers = tf.layers.flatten(layers)
            style_outputs = tf.layers.dense(layers, num_styles, activation=tf.nn.softmax)
            return style_outputs



# This implementation of the GAN looks correct and follows the general structure of a GAN. It defines the generator and discriminator networks, loads the data, computes the GAN loss and optimizer, and trains the GAN.

# However, there are a few issues with the code:

# The code uses the deprecated TensorFlow 1.x API, which is not recommended for new projects. It should be updated to use the TensorFlow 2.x API, which has a more intuitive and user-friendly interface.
# The code does not use the TensorFlow @tf.function decorator to compile the main function into a TensorFlow graph, which can improve its performance.
# The code does not save the generated images to the file system.
# The code does not handle the case where the input loss or learning rate are None, which can happen if they are not specified in the main function.
# Here is an updated version of the GAN that addresses these issues and uses the TensorFlow 2.x API:
