from collections import Counter

import numpy as np
from matplotlib import pyplot as plt

from cnn_model import CNNDREModel
import mne
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from config import edf_folder, excel_path, min_len, target_freq
from tensorflow.keras.utils import to_categorical



def get_data(edf_folder, excel_path, subtract_mean=True, subtract_axis=0,
             transpose=False, resample=True, min_len=10, target_freq=128):
    """
    Load data from .edf files, associate labels from Excel, shuffle and split
    into training, validation, and test sets. Optionally normalize and transpose.
    """

    x = []
    y = []

    df_labels = pd.read_excel(excel_path)
    label_dict = dict(zip(df_labels['new_name'], df_labels['label']))
    temp_data = []

    for filename in os.listdir(edf_folder):
        if filename.endswith('.edf') and filename in label_dict:
            filepath = os.path.join(edf_folder, filename)
            raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)
            if resample:
              raw.resample(target_freq, npad="auto")
            data = raw.get_data()  # shape: (n_channels, n_samples)
            #print("punti: ", data.shape[1])
            temp_data.append((data, label_dict[filename]))
            min_len = min(min_len, data.shape[1])

    for data, label in temp_data:
        x.append(data[:, :min_len])
        y.append(label)

    """
    for i, (data, label) in enumerate(temp_data):
        print(f"File {i}: shape = {data[:, :min_len].shape}")
    """

    print("dataset totale shape: ", np.array(x).shape)
    x = np.array(x)  # shape: (N, channels, samples)
    y = np.array(y, dtype=np.int32)

    # Optional: remove rows with NaN
    mask = ~np.any(np.isnan(x), axis=(1, 2))
    x = x[mask]
    y = y[mask]

    #split
    #1. train_val vs test
    x_temp, X_test, y_temp, y_test = train_test_split(
        x, y, test_size=0.17, random_state=0, stratify=y)

    #2. train vs val
    X_train, X_val, y_train, y_val = train_test_split(
        x_temp, y_temp, test_size=0.2, random_state=0, stratify=y_temp)
    #60% train, 20% val, 20% test (tutte bilanciate)

    # Transpose if requested
    #X_train ha forma (num_samples, timesteps, samples) verrà trasformato in (num_samples, samples, timesteps)
    print("x_train shape prima di transpose: ", np.array(X_train).shape)

    if transpose:
        X_train = np.transpose(X_train, (0, 2, 1))
        X_val = np.transpose(X_val, (0, 2, 1))
        X_test = np.transpose(X_test, (0, 2, 1))
        print("x_train shape dopo transpose: ", np.array(X_train).shape)

    # Normalize if requested
    if subtract_mean:
        for data in [X_train, X_val, X_test]:
            mean_image = np.mean(data, axis=subtract_axis, keepdims=True)
            data -= mean_image

    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
    }

def print_dataset_statistics(data):
    for split in ['train', 'val', 'test']:
        X = data[f'X_{split}']
        y = data[f'y_{split}']

        print(f"Split: {split.upper()}")
        print(f"  - Numero di campioni: {len(y)}")
        print(f"  - shape X: {X.shape}")
        print(f"  - shape y: {y.shape}")

        # Distribuzione delle classi
        if y.ndim > 1 and y.shape[1] > 1:
            # One-hot → converti a etichette
            y_labels = np.argmax(y, axis=1)
        else:
            y_labels = y.flatten()

        label_counts = Counter(y_labels)
        print("  - Distribuzione delle etichette:")
        for label, count in label_counts.items():
            perc = (count / len(y)) * 100
            print(f"    Classe {label}: {count} campioni ({perc:.1f}%)")

        print("")

        plt.figure(figsize=(4,2))
        plt.bar(label_counts.keys(), label_counts.values(), color='skyblue')
        plt.title(f'Distribuzione etichette - {split.upper()}')
        plt.xlabel('Classe')
        plt.ylabel('N')
        plt.xticks(list(label_counts.keys()))
        plt.tight_layout()
        plt.show()

def one_hot_encoding(y_train, N_val, y_val, y_test):
    print("y_train shape prima", y_train.shape)
    y_train = to_categorical(y_train, num_classes=2)
    if (N_val > 0):
        y_val = to_categorical(y_val, num_classes=2)
    y_test = to_categorical(y_test, num_classes=2)
    print("y_train shape dopo", y_train.shape)