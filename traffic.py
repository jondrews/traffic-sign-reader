import os
import sys

import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tqdm import tqdm

EPOCHS = 15
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    print(f"x_train size is {len(x_train)} | x_test size is {len(x_test)}.")
    print(f"y_train size is {len(y_train)} | y_test size is {len(y_test)}.\n")

    # Get a compiled neural network
    print("Compiling model...")
    model = get_model()

    # Fit model on training data (Train the neural network)
    print("\nTraining model...")
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    print("Model training complete. Evaluating model...")
    model.evaluate(x_test, y_test, verbose=2)
    print("Evaluation complete.")

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.
    """

    images = []  # list of numpy.ndarrays, each representing an image
    labels = []  # list of integers, corresponding to each item in images array

    print("\nLoading training data...")

    for label in tqdm(range(NUM_CATEGORIES)):
        path = os.path.join(data_dir, str(label))
        for file in os.listdir(path):
            filename = os.fsdecode(file)
            if filename.endswith(".ppm"):
                try:
                    image = cv2.imread(os.path.join(data_dir, str(label), filename))
                    images.append(cv2.resize(image, (30, 30)))
                    labels.append(label)
                except Exception as e:
                    print(e)

    print(f"Loaded {len(images)} images, with {len(labels)} labels.\n")

    return (images, labels)


def get_model():
    """
    Returns a compiled convolutional neural network model.
    """

    # Create a convolutional neural network
    model = tf.keras.models.Sequential(
        [
            # Convolutional layer.
            tf.keras.layers.Conv2D(
                3, (3, 3), activation="relu", input_shape=(30, 30, 3)
            ),

            # Convolutional layer.
            tf.keras.layers.Conv2D(9, (3, 3), activation="relu"),

            # Convolutional layer.
            tf.keras.layers.Conv2D(27, (3, 3), activation="relu"),

            # Max-pooling layer, using 2x2 pool size
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

            # Flatten units into one dimension
            tf.keras.layers.Flatten(),

            # Add a hidden layer with dropout
            tf.keras.layers.Dense(128, activation="sigmoid"),
            tf.keras.layers.Dropout(0.5),

            # Add an output layer with output units for all 43 categories
            tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax"),
        ]
    )

    # Compile neural network
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    return model


if __name__ == "__main__":
    main()
