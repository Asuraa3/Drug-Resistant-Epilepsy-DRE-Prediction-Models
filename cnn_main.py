import numpy as np
from cnn_model import CNNDREModel
import mne
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from config import edf_folder, excel_path, min_len, target_freq, num_classes, num_epochs
import data_processing


def main():

    l = os.listdir(edf_folder)
    print(len(l))


    data = data_processing.get_data(edf_folder, excel_path, subtract_mean=True, subtract_axis=1,
                    transpose=True, resample=False, min_len=min_len, target_freq=target_freq)

    print("")
    print("il dizionario è fatto di:")
    for k in data.keys():
        print('{}: {} '.format(k, data[k].shape))

    data_processing.print_dataset_statistics(data)

    # substract data from list
    X_train = data.get('X_train')
    y_train = data.get('y_train')
    X_val = data.get('X_val')
    y_val = data.get('y_val')
    X_test = data.get('X_test')
    y_test = data.get('y_test')

    # get data dimension
    N_train, T_train, C_train = data.get('X_train').shape
    N_val, T_val, C_val = data.get('X_val').shape
    N_test, T_test, C_test = data.get('X_test').shape

    sampling = 1  # non modifica niente

    X_train = X_train.reshape(N_train, int(T_train / sampling), sampling, C_train)[:, :, 0, :]
    X_val = X_val.reshape(N_val, int(T_val / sampling), sampling, C_val)[:, :, 0, :]
    X_test = X_test.reshape(N_test, int(T_test / sampling), sampling, C_test)[:, :, 0, :]

    # get new data dimension
    N_train, T_train, C_train = X_train.shape
    N_val, T_val, C_val = X_val.shape
    N_test, T_test, C_test = X_test.shape

    print('X_train: ', X_train.shape)
    print('y_train: ', y_train.shape)
    print('X_val: ', X_val.shape)
    print('y_val: ', y_val.shape)
    print('X_test: ', X_test.shape)
    print('y_test: ', y_test.shape)

    channels = C_train
    timesteps = T_train
    batch_size = int(X_train.shape[0] / 2)  # 59
    print("batch_size: ", batch_size)



    model = CNNDREModel(n_channels=channels, n_timepoints=timesteps, batch_size=batch_size)
    history = model.train_eeg_model(X_train, y_train, X_val, y_val, epochs=num_epochs)
    print("Valutazione sul test set:")
    results = model.eeg_model.evaluate(X_test, y_test, verbose=2)
    print(f"Test loss: {results[0]:.4f}, Test accuracy: {results[1]:.4f}")

if __name__ == "__main__":
    main()
