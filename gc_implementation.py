# Author: Luke Patterson
# Name: gc_implementation.py
# Description: This file contains code for implementing Grad-CAM on the pre-interview task.
#   taking inspiration from https://deeplearningofpython.blogspot.com/2023/05/Gradcam-working-example-python.html

import cv2
import numpy as np
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
from preprocessing import resize_and_normalize
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


# Function to create a heatmap for a given image using Grad-CAM algorithm
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Create a heatmap for a given image using Grad-CAM algorithm.
    :param img_array: The image as a numpy array.
    :param model: The model.
    :param last_conv_layer_name: The name of the last convolutional layer.
    :param pred_index: The index of the predicted class.
    :return: The heatmap as a numpy array.
    """
    # Create a model that maps the input image to the activations of the last conv layer
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Determine the index of the class with the highest predicted probability
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # Compute the gradient of the relevant class
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # Pool the gradients across the channels
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array by "how important this channel is"
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize the heatmap between 0 and 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


# Function to overlay the heatmap on the original image
def overlay_heatmap(heatmap, image, alpha=0.4):
    """
    Overlay the heatmap on the original image.
    :param heatmap: The heatmap as a numpy array.
    :param image: The image as a numpy array.
    :param alpha: The alpha value for the heatmap.
    :return: The image with the heatmap overlaid.
    """
    # Resize the heatmap to match the image size
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

    # Convert the heatmap to RGB
    heatmap_resized = np.uint8(255 * heatmap_resized)
    heatmap_resized = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)

    # Convert the grayscale image to a three-channel image
    if len(image.shape) == 2 or image.shape[2] == 1:
        image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2RGB)

    # Overlay the heatmap on the original image
    superimposed_img = heatmap_resized * alpha + image
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    return superimposed_img


if __name__ == '__main__':
    # Load model and test data
    model_path = 'models/model_20231210-194623_F1_0_5953.h5'
    model = tf.keras.models.load_model(model_path)

    # Specify the last convolutional layer
    last_conv_layer_name = "conv_7b_ac"

    # Load and preprocess images for visualization
    with open('split_data/test.pkl', 'rb') as f:
        test = pickle.load(f)
    X_test, y_test = test

    # Select images for visualization
    selected_images = []
    selected_originals = []  # Store the original images
    for label in np.unique(y_test):
        idx = np.where(y_test == label)[0][0]  # Taking first instance
        selected_originals.append(X_test[idx])  # Save original spectrogram
        processed_img = resize_and_normalize(np.array([X_test[idx]]))[0]  # Process for model input
        selected_images.append(processed_img)

    # Process and visualize
    for idx, (orig_img, img) in enumerate(zip(selected_originals, selected_images)):
        heatmap = make_gradcam_heatmap(np.expand_dims(img, axis=0), model, last_conv_layer_name)
        superimposed_img = overlay_heatmap(heatmap, orig_img)

        # Display original image
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(orig_img, cmap='gray')
        plt.title(f'Original - {idx}')
        plt.axis('off')

        # Display Grad-CAM image
        plt.subplot(1, 2, 2)
        heatmap_img = plt.imshow(superimposed_img)
        plt.title(f'Grad-CAM - {idx}')
        plt.axis('off')

        # Save the figure
        plt.savefig(f'gradcam_images/gradcam_{idx}.png')

