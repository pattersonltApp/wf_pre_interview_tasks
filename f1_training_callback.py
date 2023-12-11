# Author: Luke Patterson
# Name: f1_training_callback.py
# Description: This file contains code for a custom callback to log F1 scores and save model during training.

import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
import wandb
from sklearn.metrics import confusion_matrix
import seaborn as sns
import datetime
import gc


class PhysioNetF1Callback(keras.callbacks.Callback):
    def __init__(self, validation_data, class_names, batch_size=16):
        super().__init__()
        self.validation_data = validation_data
        self.best_f1_score = -1  # Initialize with negative value to ensure the first epoch's score is saved
        self.class_names = class_names
        self.batch_size = batch_size

    def on_epoch_end(self, epoch, logs=None):
        """
        Calculate the F1 score for each class and the average F1 score (the final challenge score).
        :param epoch: The current epoch.
        :param logs: The logs.
        :return: None, but saves the model and a confusion matrix if the F1 score has improved.
        """
        # Initialize predictions and true values
        val_preds = []
        val_trues = []

        # Predict in batches
        for i in range(0, len(self.validation_data[0]), self.batch_size):
            batch_x = self.validation_data[0][i:i + self.batch_size]
            batch_y = self.validation_data[1][i:i + self.batch_size]
            batch_preds = self.model.predict(batch_x)
            val_preds.extend(np.argmax(batch_preds, axis=1))
            val_trues.extend(np.argmax(batch_y, axis=1))

        # Convert to numpy arrays
        val_preds = np.array(val_preds)
        val_trues = np.array(val_trues)

        # Calculate confusion matrix
        cm = confusion_matrix(val_trues, val_preds)

        # Calculate F1 scores for each class using PhysioNet's method
        F1_scores = {}
        for i in range(len(cm)):
            F1 = 2 * cm[i, i] / (np.sum(cm[i, :]) + np.sum(cm[:, i]) if np.sum(cm[i, :]) + np.sum(cm[:, i]) > 0 else 0)
            F1_scores[f'F1_{i}'] = F1

        # Calculate the average F1 score (the final challenge score)
        avg_F1_score = np.mean(list(F1_scores.values()))
        F1_scores['avg_F1_score'] = avg_F1_score

        # Log F1 scores
        wandb.log(F1_scores)

        # Save the model if the F1 score has improved
        if avg_F1_score > self.best_f1_score:
            self.best_f1_score = avg_F1_score
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            avg_F1_formatted = "{:.4f}".format(avg_F1_score).replace('.', '_')
            filename = f'model_{timestamp}_F1_{avg_F1_formatted}.h5'
            self.model.save(f'models/{filename}')

            # # Generate and log the confusion matrix image
            fig = self.plot_confusion_matrix(cm, filename=filename, class_names=self.class_names)
            wandb.log({"confusion_matrix": [wandb.Image(fig, caption="Confusion Matrix")]})
            plt.close(fig)

        # Clean up, might not be necessary now that batch processing is implemented
        del val_preds, val_trues, cm, F1_scores, avg_F1_score
        gc.collect()

    @staticmethod
    def plot_confusion_matrix(cm, filename, class_names):
        """
        Returns a figure containing the plotted confusion matrix.
        :param cm: The confusion matrix.
        :param filename: The filename to save the figure as.
        :param class_names: The class names.
        :return: The figure.
        """
        figure = plt.figure(figsize=(8, 8))
        ax = sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        ax.xaxis.tick_top()
        plt.title(filename)
        plt.ylabel('Reference Classification')
        plt.xlabel('Predicted Classification')
        return figure
