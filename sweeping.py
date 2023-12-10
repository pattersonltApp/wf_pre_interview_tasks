# Author: Luke Patterson
# Name: sweeping.py
# Description: This file contains code for a WandB sweep for the pre-interview task.

import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras
import wandb
from wandb.keras import WandbMetricsLogger
from sklearn.metrics import confusion_matrix

# Login to wandb
wandb.login()

sweep_config = {
    'method': 'bayes',
    'metric': {
        'name': 'avg_F1_score',
        'goal': 'maximize'
    },
    'parameters': {
        'num_neurons': {
            'values': [128, 256, 512, 1024]
        },
        'activation': {
            'values': ['relu', 'tanh', 'sigmoid']
        },
        'learning_rate': {
            'min': 0.0001,
            'max': 0.01
        },
        'batch_size': {
            'values': [16, 32]
        },
        'optimizer': {
            'values': ['adam', 'sgd', 'rmsprop']
        }
    }
}


# Log the hardware used
# gpus = tf.config.list_physical_devices('GPU')
# print("Num GPUs Available: ", len(gpus))
# if gpus:
#     try:
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#     except RuntimeError as e:
#         print(e)

# Define custom callback to log F1 scores
class PhysioNetF1Callback(keras.callbacks.Callback):
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


def get_optimizer(name, learning_rate):
    if name == 'adam':
        return tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif name == 'sgd':
        return tf.keras.optimizers.SGD(learning_rate=learning_rate)
    elif name == 'rmsprop':
        return tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    else:
        raise ValueError("Unsupported optimizer name")


def training():
    tf.keras.backend.clear_session()
    with wandb.init() as run:
        config = run.config
        # Load the InceptionResNetV2 model
        base_model = keras.applications.inception_resnet_v2.InceptionResNetV2(
          weights='imagenet', include_top=False, input_shape=(299, 299, 3))
        base_model.trainable = False

        # Add custom layers
        x = keras.layers.GlobalAveragePooling2D()(base_model.output)
        x = keras.layers.Dense(config.num_neurons, activation='relu')(x)
        output = keras.layers.Dense(len(class_names), activation='softmax')(x)

        model = keras.Model(inputs=base_model.input, outputs=output)

        # Compile the model with optimizer and learning rate from config
        optimizer = get_optimizer(config.optimizer, config.learning_rate)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        # Create an instance of the custom F1 callback
        f1_callback = PhysioNetF1Callback(validation_data=(X_val, y_val_categorical), class_names=class_names)

        num_epochs = 10

        wandb_callbacks = [
            f1_callback,
            WandbMetricsLogger()
        ]

        # Train the model with both the WandbCallback and your custom F1 callback
        model.fit(
          X_train, y_train_categorical,
          validation_data=(X_val, y_val_categorical),
          epochs=num_epochs,
          batch_size=config.batch_size,
          callbacks=wandb_callbacks
        )


with open('split_data/train.pkl', 'rb') as f:
    train = pickle.load(f)
with open('split_data/val.pkl', 'rb') as f:
    val = pickle.load(f)

# train, test, val should all be (X, y) tuples
X_train, y_train = train
X_val, y_val = val

# Resize and normalize the data for InceptionResNetV2
X_train = resize_and_normalize(X_train)
X_val = resize_and_normalize(X_val)

# Encode labels to integers
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_val_encoded = le.transform(y_val)

# Store the class names from the LabelEncoder
class_names = le.classes_.tolist()

# Convert labels to one-hot encoding
y_train_categorical = keras.utils.to_categorical(y_train_encoded)
y_val_categorical = keras.utils.to_categorical(y_val_encoded)

sweep_id = wandb.sweep(sweep_config, project='pre_interview_tasks', entity='pattersonlt')
wandb.agent(sweep_id, training)


