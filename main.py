# Author: Luke Patterson
# Name: main.py - ECG Classification - Pre-Interview Tasks
# Description: This file contains the code for pre-interview tasks involving ECG Classification.

import pandas as pd
import scipy.io as sio
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import spectrogram
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras
import wandb
from wandb.keras import WandbMetricsLogger
from f1_training_callback import PhysioNetF1Callback
from preprocessing import resize_and_normalize

# Login to wandb
wandb.login()

# Initialize WandB with the project configuration
wandb.init(project='pre_interview_tasks', entity='pattersonlt', config={
    "learning_rate": 0.003,
    "epochs": 100,
    "batch_size": 16,
    "num_neurons": 500,
    "activation": "sigmoid"
})

# Log the hardware used and set memory growth for GPU, this should help with memory issues
gpus = tf.config.list_physical_devices('GPU')
print("Num GPUs Available: ", len(gpus))
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


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


def plot_sequence_length_distribution(df):
    """
    Plots the distribution of sequence lengths in the DataFrame.
    :param df: DataFrame with ECG data and class labels
    :return: None (saves plot to file)
    """
    # Convert sample lengths to seconds (sampling rate is 300 Hz for all data)
    lengths_in_seconds = df['ecg_data'].apply(len) / 300

    # Define bins for the histogram
    bins = [0, 10, 20, 30, 40, 50, 60, max(lengths_in_seconds)]

    # Create a histogram
    plt.figure(figsize=(12, 6))
    plt.hist(lengths_in_seconds, bins=bins, alpha=0.75, color='gold', edgecolor='black')

    plt.xlabel('Length of Sequence (seconds)')
    plt.ylabel('Count of Sequences')
    plt.title('Distribution of ECG Data Sequence Lengths in Seconds')
    plt.xlim(0, 70)
    plt.savefig('data_exploration/sequence_length_distribution.png')
    print(f'Sequence length distribution plot saved to data_exploration directory.')


def plot_class_distribution(df):
    """
    Plots the distribution of classes in the DataFrame.
    :param df: DataFrame with ECG data and class labels
    :return: None (saves plot to file)
    """
    # Count the number of collections in each class
    class_counts = df['class_label'].value_counts()

    # Create a bar plot
    plt.figure(figsize=(10, 6))
    bar_plot = class_counts.plot(kind='bar', color='gold', edgecolor='black')

    plt.xlabel('Class Label')
    plt.ylabel('Count of Recordings')
    plt.title('Count of ECG Recordings for Each Class')
    plt.xticks(rotation=0)

    # Annotate the bars with the counts
    for i, value in enumerate(class_counts):
        plt.text(i, value, str(value), ha='center', va='bottom')

    plt.savefig('data_exploration/class_distribution.png')
    print(f'Class distribution plot saved to data_exploration directory.')


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


def load_and_train_model(X_train, y_train_categorical, X_val, y_val_categorical, class_names):
    """
    Load the InceptionResNetV2 model, adapt it to the ECG task, and train.
    :param X_train: The training data.
    :param y_train_categorical: The training labels.
    :param X_val: The validation data.
    :param y_val_categorical: The validation labels.
    :param class_names: The class names.
    :return model: The trained model.
    """
    tf.keras.backend.clear_session()
    # Load the InceptionResNetV2 model
    base_model = keras.applications.inception_resnet_v2.InceptionResNetV2(
        weights='imagenet', include_top=False, input_shape=(299, 299, 3))
    base_model.trainable = False  # Freeze base model layers

    # Add custom layers
    x = keras.layers.GlobalAveragePooling2D()(base_model.output)
    x = keras.layers.Dense(wandb.config.num_neurons, activation=wandb.config.activation)(x)
    output = keras.layers.Dense(len(class_names), activation='softmax')(x)

    model = keras.Model(inputs=base_model.input, outputs=output)

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=wandb.config.learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

    # Create an instance of the custom F1 callback
    f1_callback = PhysioNetF1Callback(validation_data=(X_val, y_val_categorical), class_names=class_names)

    # Train the model with both the WandbCallback and your custom F1 callback
    model.fit(
        X_train, y_train_categorical,
        validation_data=(X_val, y_val_categorical),
        epochs=wandb.config.epochs,
        batch_size=wandb.config.batch_size,
        callbacks=[f1_callback, WandbMetricsLogger()]
    )

    return model


if __name__ == '__main__':
    # Create or load DataFrame
    if not os.path.exists('ecg_data.pkl'):
        df = create_df()
    else:
        df = pd.read_pickle('ecg_data.pkl')

    # If sequence length distribution plot does not exist, create it
    if not os.path.exists('data_exploration/sequence_length_distribution.png'):
        plot_sequence_length_distribution(df)

    # If class distribution plot does not exist, create it
    if not os.path.exists('data_exploration/class_distribution.png'):
        plot_class_distribution(df)

    # If split data does not exist, create it, otherwise load it
    if not os.path.exists('split_data/train.pkl'):
        train, test, val = prepare_and_pickle_data(df)
        del test  # Don't need test set
    else:
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
    num_classes = len(le.classes_)
    model = load_and_train_model(X_train, y_train_categorical, X_val, y_val_categorical, class_names)
