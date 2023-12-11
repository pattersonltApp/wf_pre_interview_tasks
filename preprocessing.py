# Author: Luke Patterson
# Name: preprocessing.py
# Description: This file contains code for preprocessing the data that might be used across multiple files.

import numpy as np
import tensorflow as tf
from tensorflow import keras


def resize_and_normalize(X, target_size=(299, 299)):
    """
    Resize the spectrograms to the target size and normalize them for InceptionResNetV2.
    :param X: The spectrograms.
    :param target_size: The target size.
    :return X_normalized: The normalized spectrograms.
    """
    X_resized = [tf.image.resize(tf.expand_dims(img, axis=-1), target_size) for img in X]  # Add channel dimension
    X_three_channel = [tf.repeat(img, 3, axis=-1) for img in X_resized]  # Convert to 3-channel
    X_normalized = np.array([keras.applications.inception_resnet_v2.preprocess_input(x) for x in X_three_channel])
    return X_normalized