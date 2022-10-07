import cv2
import numpy as np
import os
import sys
import tensorflow as tf
# import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
DATA_DIR = 'gtsrb-small'  # TODO: change to 'gtsrb'
NUM_CATEGORIES = 3  # TODO: change me back to 43
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
        if len(sys.argv) not in [2, 3]:
            sys.exit("Usage: python traffic.py data_directory [model.h5]")

        # Get image arrays and labels for all image files
        images, labels = load_data(sys.argv[1])
        print(f"Received {len(images)} images, and {len(labels)} labels from load_data().\n")

        # Split data into training and testing sets
        labels = tf.keras.utils.to_categorical(labels)
        x_train, x_test, y_train, y_test = train_test_split(
            np.array(images), np.array(labels), test_size=TEST_SIZE
        )
        print(f"x_train size is {len(x_train)}.")
        print(f"x_test size is {len(x_test)}.")
        print(f"y_train size is {len(y_train)}.")
        print(f"y_test size is {len(y_test)}.\n")

        """    
        # Get a compiled neural network
        model = get_model()

        # Fit model on training data
        model.fit(x_train, y_train, epochs=EPOCHS)

        # Evaluate neural network performance
        model.evaluate(x_test,  y_test, verbose=2)

        # Save model to file
        if len(sys.argv) == 3:
            filename = sys.argv[2]
            model.save(filename)
            print(f"Model saved to {filename}.")

    """


def load_data(data_dir=DATA_DIR):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    --------
    The load_data function should accept as an argument data_dir, representing the path to a directory where the data is stored, 
    and return image arrays and labels for each image in the data set.

    You may assume that data_dir will contain one directory named after each category, numbered 0 through NUM_CATEGORIES - 1. 
    Inside each category directory will be some number of image files.
    Use the OpenCV-Python module (cv2) to read each image as a numpy.ndarray (a numpy multidimensional array). 
    To pass these images into a neural network, the images will need to be the same size, so be sure to resize each image to have 
    width IMG_WIDTH and height IMG_HEIGHT.
    The function should return a tuple (images, labels). images should be a list of all of the images in the data set, where each 
    image is represented as a numpy.ndarray of the appropriate size. labels should be a list of integers, representing the category 
    number for each of the corresponding images in the images list.
    Your function should be platform-independent: that is to say, it should work regardless of operating system. 
    Note that on macOS, the / character is used to separate path components, while the \ character is used on Windows. 
    Use os.sep and os.path.join as needed instead of using your platform's specific separator character.
    """

    images = []  # each image is a numpy.ndarray of appropriate size
    labels = []  # list of integers (category), corresponding to each item in images array

    for label in range(NUM_CATEGORIES):
        # print(f"Loading images for label {label}...")
        path = os.path.join(data_dir, str(label))
        for file in os.listdir(path):
            filename = os.fsdecode(file)
            if filename.endswith(".ppm"):
                try:
                    # print(f"   loading {os.path.join(data_dir, str(label), filename)}")
                    image = cv2.imread(os.path.join(data_dir, str(label), filename))
                    images.append(cv2.resize(image, (30, 30)))
                    labels.append(label)
                except Exception as e:
                    print(e)

    print(f"\nProcessed {len(images)} images, and {len(labels)} labels.")

    return (images, labels)


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    The get_model function should return a compiled neural network model.

    You may assume that the input to the neural network will be of the shape (IMG_WIDTH, IMG_HEIGHT, 3) (that is, an 
    array representing an image of width IMG_WIDTH, height IMG_HEIGHT, and 3 values for each pixel for red, green, and blue).
    The output layer of the neural network should have NUM_CATEGORIES units, one for each of the traffic sign categories.
    The number of layers and the types of layers you include in between are up to you. You may wish to experiment with:
        different numbers of convolutional and pooling layers
        different numbers and sizes of filters for convolutional layers
        different pool sizes for pooling layers
        different numbers and sizes of hidden layers
        dropout
    """
    raise NotImplementedError


if __name__ == "__main__":
    main()
