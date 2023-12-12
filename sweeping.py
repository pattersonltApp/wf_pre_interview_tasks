# Author: Luke Patterson
# Name: sweeping.py
# Description: This file contains code for a WandB sweep for the pre-interview task.

import pickle
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras
import wandb
from wandb.keras import WandbMetricsLogger
from f1_sweeping_callback import PhysioNetF1SweepCallback
from preprocessing import resize_and_normalize

# Login to wandb
wandb.login()

# Sweep config to be used by WandB
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
gpus = tf.config.list_physical_devices('GPU')
print("Num GPUs Available: ", len(gpus))
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


def get_optimizer(name, learning_rate):
    """
    Returns an optimizer based on the name and learning rate.
    :param name: The name of the optimizer.
    :param learning_rate: The learning rate to use.
    :return: An optimizer.
    """
    if name == 'adam':
        return tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif name == 'sgd':
        return tf.keras.optimizers.SGD(learning_rate=learning_rate)
    elif name == 'rmsprop':
        return tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    else:
        raise ValueError("Unsupported optimizer name")


def training():
    """
    This function is called by WandB to train the model.
    :return: None
    """
    tf.keras.backend.clear_session()
    with wandb.init() as run:
        config = run.config
        # Load the InceptionResNetV2 model
        base_model = keras.applications.inception_resnet_v2.InceptionResNetV2(
          weights='imagenet', include_top=False, input_shape=(299, 299, 3))
        base_model.trainable = False

        # Add custom layers
        x = keras.layers.GlobalAveragePooling2D()(base_model.output)
        x = keras.layers.Dense(config.num_neurons, activation=config.activation)(x)
        output = keras.layers.Dense(len(class_names), activation='softmax')(x)

        model = keras.Model(inputs=base_model.input, outputs=output)

        # Compile the model with optimizer and learning rate from config
        optimizer = get_optimizer(config.optimizer, config.learning_rate)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        # Create an instance of the custom F1 callback
        f1_callback = PhysioNetF1SweepCallback(validation_data=(X_val, y_val_categorical), class_names=class_names)

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


# Load data
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

# Initialize the sweep
sweep_id = wandb.sweep(sweep_config, project='pre_interview_tasks', entity='pattersonlt')
wandb.agent(sweep_id, training)
