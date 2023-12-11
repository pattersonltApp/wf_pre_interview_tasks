# Author: Luke Patterson
# Name: f1_sweeping_callback.py
# Description: This file contains code for a custom callback to log F1 scores during a sweep.

import numpy as np
from tensorflow import keras
import wandb
from sklearn.metrics import confusion_matrix


# Define custom callback to log F1 scores during sweep.
class PhysioNetF1SweepCallback(keras.callbacks.Callback):
    def __init__(self, validation_data, class_names):
        super().__init__()
        self.validation_data = validation_data
        self.best_f1_score = -1
        self.class_names = class_names

    def on_epoch_end(self, epoch, logs=None):
        """
        Calculate the F1 score for each class and the average F1 score (the final challenge score).
        :param epoch: The current epoch.
        :param logs: The logs.
        :return: None
        """
        # Predict the validation data
        val_pred_raw = self.model.predict(self.validation_data[0])
        val_pred = np.argmax(val_pred_raw, axis=1)
        val_true = np.argmax(self.validation_data[1], axis=1)

        # Calculate confusion matrix
        cm = confusion_matrix(val_true, val_pred)

        # Calculate F1 scores for each class using PhysioNet's method
        F1_scores = {}
        for i in range(len(cm)):
            F1 = 2 * cm[i, i] / (np.sum(cm[i, :]) + np.sum(cm[:, i]) if np.sum(cm[i, :]) + np.sum(cm[:, i]) > 0 else 1)
            F1_scores[f'F1_{i}'] = F1

        # Calculate the average F1 score (the final challenge score)
        avg_F1_score = np.mean(list(F1_scores.values()))
        F1_scores['avg_F1_score'] = avg_F1_score

        # Log F1 scores
        wandb.log(F1_scores)
