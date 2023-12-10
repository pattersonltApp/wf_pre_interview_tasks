# Author: Luke Patterson
# Name: ECG Classification - Pre-Interview Tasks
# Description: This file contains the code for pre-interview tasks involving ECG Classification.

import pandas as pd
import scipy.io as sio
import os
import pickle
import numpy as np
from scipy.signal import spectrogram
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras
import wandb
from wandb.keras import WandbCallback
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
            'values': [16, 32, 64]
        },
        'optimizer': {
            'values': ['adam', 'sgd', 'rmsprop']
        },
        'num_dense_layers': {
            'values': [1, 2, 3]
        }
    }
}


# Log the hardware used
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


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


def create_df():
    """
    Creates a DataFrame from the .mat files in the training2017 folder and pickles it for later use.
    This function should only be run once. After the DataFrame is created, it can be loaded from the pickle file.
    This is done to avoid having to read the .mat files every time the program is run.
    :return: DataFrame with ECG data and class labels
    """
    # Read the REFERENCE.csv file to extract the class labels
    ref_df = pd.read_csv('training2017/REFERENCE.csv', header=None, names=['name', 'class_label'])
    ref_dict = dict(zip(ref_df['name'], ref_df['class_label']))

    # Iterate through .mat files and create DataFrame with ECG data and class labels
    data = []
    for file in os.listdir('training2017'):
        if file.endswith('.mat'):
            file_path = os.path.join('training2017', file)
            mat_data = sio.loadmat(file_path)
            ecg_data = mat_data['val'][0]
            file_name = file.split('.')[0]
            class_label = ref_dict[file_name]
            data.append({'name': file_name, 'ecg_data': ecg_data, 'class_label': class_label})

    df = pd.DataFrame(data)

    # Pickle the DataFrame for later use
    pickle_file = 'ecg_data.pkl'
    with open(pickle_file, 'wb') as f:
        pickle.dump(df, f)

    print(f'DataFrame pickled to {pickle_file}')
    return df


def preprocess_ecg_data(ecg_sequence, target_length=18000):
    """
    Preprocess the ECG data to a fixed length by padding or truncating. Then compute the spectrogram.
    :param ecg_sequence: The ECG data sequence.
    :param target_length: The target length of the sequence.
    :return spectrogram_data: The spectrogram data.
    """
    current_length = len(ecg_sequence)
    if current_length < target_length:
        # Pad the sequence with zeros
        ecg_sequence = np.pad(ecg_sequence, (0, target_length - current_length), mode='constant')
    elif current_length > target_length:
        # Truncate the sequence
        ecg_sequence = ecg_sequence[:target_length]

    # Compute the spectrogram
    # Note: Using fs//2 and fs//4 for nperseg and noverlap respectively, not sure if this is best. Experiment with this.
    _, _, spectrogram_data = spectrogram(ecg_sequence, fs=300, nperseg=150, noverlap=75)

    # Apply log transformation,
    log_spectrogram_data = np.log1p(spectrogram_data)
    return log_spectrogram_data


def prepare_and_pickle_data(df):
    """
    Prepares the data by splitting it into train, test, and validation sets and then pickles the sets.
    :param df: The ECG DataFrame.
    :return train, test, val: The train, test, and validation sets. Also saves the sets to pickle files.
    """
    # Create X and y
    X = [preprocess_ecg_data(sequence) for sequence in df['ecg_data'].tolist()]
    y = df['class_label']

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Further split the training set into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, stratify=y_train, random_state=42)

    # Combine datasets for train, test, and validation
    combined_datasets = {
        'train': (X_train, y_train),
        'test': (X_test, y_test),
        'val': (X_val, y_val)
    }

    for name, data in combined_datasets.items():
        with open(f'split_data/{name}.pkl', 'wb') as f:
            pickle.dump(data, f)

    print(f'Data successfully split and pickled in split_data directory.')
    return combined_datasets['train'], combined_datasets['test'], combined_datasets['val']


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


def load_and_train_model(config, X_train, y_train, X_val, y_val, class_names):
    """
    Load the InceptionResNetV2 model, adapt it to the ECG task, and train.
    :param X_train: The training data.
    :param y_train: The training labels.
    :param X_val: The validation data.
    :param y_val: The validation labels.
    :param class_names: The class names.
    :return model: The trained model.
    """
    # Load the InceptionResNetV2 model
    base_model = keras.applications.inception_resnet_v2.InceptionResNetV2(
        weights='imagenet', include_top=False, input_shape=(299, 299, 3))
    base_model.trainable = False

    # Add custom layers
    x = keras.layers.GlobalAveragePooling2D()(base_model.output)
    for _ in range(config.num_dense_layers):
        x = keras.layers.Dense(config.num_neurons, activation=config.activation)(x)
    output = keras.layers.Dense(len(class_names), activation='softmax')(x)

    model = keras.Model(inputs=base_model.input, outputs=output)

    # Compile the model with optimizer and learning rate from config
    optimizer = get_optimizer(config.optimizer, config.learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # Create an instance of the custom F1 callback
    f1_callback = PhysioNetF1Callback(validation_data=(X_val, y_val), class_names=class_names)

    num_epochs = 15
    # Train the model with both the WandbCallback and your custom F1 callback
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=num_epochs,
        batch_size=config.batch_size,
        callbacks=[f1_callback, WandbCallback()]
    )


def training():
    """
    Wrapper function to use for WandB sweeps.
    """
    # Initialize a WandB run
    with wandb.init() as run:
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

        # Load and train the model
        load_and_train_model(run.config, X_train, y_train_categorical, X_val, y_val_categorical, class_names)


if __name__ == '__main__':
    # Create or load DataFrame
    if not os.path.exists('ecg_data.pkl'):
        df = create_df()
    else:
        df = pd.read_pickle('ecg_data.pkl')

    # If split data does not exist, create it, otherwise load it
    if not os.path.exists('split_data/test.pkl'):
        train, test, val = prepare_and_pickle_data(df)

    sweep_id = wandb.sweep(sweep_config, project='pre_interview_tasks', entity='pattersonlt')
    wandb.agent(sweep_id, training)


