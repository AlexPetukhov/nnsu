import keras
from keras.losses import SparseCategoricalCrossentropy
import keras.layers as L
import os
import numpy as np
import scipy.io as sio
import pandas as pd

filter_means = np.array([32, 32, 256, 256, 256, 512, 512, 512, 512])
filter_means_2 = filter_means


def get_conv_model(filter_means, input_layer):
    for i in range(len(filter_means)):
        if i == 0:
            conv = L.Conv1D(filters=filter_means[i], kernel_size=9, padding='same', strides=5, activation='relu')(
                input_layer)
        else:
            conv = L.Conv1D(filters=filter_means[i], kernel_size=9, padding='same', strides=5, activation='relu')(relu)
        drop = L.Dropout(0.25)(conv)
        conv = L.Conv1D(filters=filter_means[i], kernel_size=9, padding='same', strides=5)(drop)
        batch = L.BatchNormalization()(conv)
        relu = L.ReLU()(batch)
        drop = L.Dropout(0.25)(relu)
    return L.Flatten()(drop)


def model_many_diseases2():
    input_layer = L.Input(shape=(4500, 12))
    input_layer2 = L.Input(shape=(4500, 4))
    input_layer3 = L.Input(shape=(60,))

    model_12 = get_conv_model(filter_means[:2], input_layer)
    model_4 = get_conv_model(filter_means_2[:2], input_layer2)

    merged = L.Concatenate(axis=1)([model_12, model_4, input_layer3])

    dense_out1 = L.Dense(512, activation='relu')(merged)
    dense_out2 = L.Dense(9)(dense_out1)

    model = keras.Model([input_layer, input_layer2, input_layer3], [dense_out2])
    return model


def get_features_from_signal(signal):
    signal = signal.T
    features = np.empty(60)
    features[:12] = signal.max(axis=1)
    features[12:24] = signal.min(axis=1)
    features[24:36] = signal.mean(axis=1)
    features[36:48] = signal.std(axis=1)
    features[48:] = signal.max(axis=1) + signal.min(axis=1)
    return features


def generator_st(x_train, y_train, batch_size):
    # Create empty arrays to contain batch of features and label

    batch_features_1 = np.zeros((batch_size, 4500, 12))
    batch_features_2 = np.zeros((batch_size, 4500, 4))
    batch_features_3 = np.zeros((batch_size, 60))
    batch_labels = np.zeros(batch_size, dtype=int)

    while True:
        for i in range(batch_size):
            # choose random index in features
            index = np.random.choice(len(x_train))
            x = x_train[index][:, :12]
            delin = x_train[index][:, 12:]
            shift = np.random.randint(0, len(x) - 4500 + 1, 1)[0]

            batch_features_1[i] = x[shift: shift + 4500]
            batch_features_2[i] = delin[shift: shift + 4500]
            batch_features_3[i, :12] = batch_features_1[i].max(axis=0)
            batch_features_3[i, 12:24] = batch_features_1[i].min(axis=0)
            batch_features_3[i, 24:36] = batch_features_1[i].mean(axis=0)
            batch_features_3[i, 36:48] = batch_features_1[i].std(axis=0)
            batch_features_3[i, 48:] = batch_features_1[i].max(axis=0) + batch_features_1[i].min(axis=0)
            batch_labels[i] = y_train[index]
        yield [batch_features_1, batch_features_2, batch_features_3], batch_labels


def load_data(record_path, delineation_path):
    data = sio.loadmat(record_path)
    signal = data['ECG']['data'][0][0]
    delineation = np.load(delineation_path) / 5
    return signal.T, delineation.T


def get_train_data(data_and_delin_path, mat_folder, delin_folder):
    data_path = os.path.join(data_and_delin_path, mat_folder)
    data_list = os.listdir(data_path)
    delin_path = os.path.join(data_and_delin_path, delin_folder)
    delin_list = os.listdir(delin_path)
    if 'REFERENCE.csv' in data_list:
        data_list.remove('REFERENCE.csv')
    if '.DS_Store' in data_list:
        data_list.remove('.DS_Store')
    if 'REFERENCE.csv' in delin_list:
        delin_list.remove('REFERENCE.csv')
    if '.DS_Store' in delin_list:
        delin_list.remove('.DS_Store')
    data_list.sort()
    delin_list.sort()

    x_train = []
    y_train = []
    reference_dict = {}
    reference_path = os.path.join(data_path, 'REFERENCE.csv')
    reader = pd.read_csv(reference_path)
    for i, row in reader.iterrows():
        reference_dict[row['Recording']] = int(row['First_label'])

    for data, delin in zip(data_list, delin_list):
        cur_data_path = os.path.join(data_path, data)
        cur_delin_path = os.path.join(delin_path, delin)
        signal, delineation = load_data(cur_data_path, cur_delin_path)
        x_train.append(np.concatenate((signal, delineation), axis=1))
        data_name = data.split('.')[0]
        y_train.append(reference_dict[data_name] - 1)
    return x_train, y_train


def main():
    data_and_delin_path = 'DATA'
    mat_folder = 'TrainingSet1'
    delin_folder = 'delineation_leads_val'
    x_train, y_train = get_train_data(data_and_delin_path, mat_folder, delin_folder)

    X_val = np.load(os.path.join(data_and_delin_path, 'X.npy'))
    y_val = np.load(os.path.join(data_and_delin_path, 'y.npy'))
    name_val = np.load(os.path.join(data_and_delin_path, 'name.npy'))
    X_features = np.array([get_features_from_signal(x) for x in X_val[:, :, :12]])

    model = model_many_diseases2()

    model.compile(loss=SparseCategoricalCrossentropy(from_logits=True), optimizer='adam', metrics=['accuracy'])

    history = model.fit(generator_st(x_train=x_train, y_train=y_train, batch_size=128),
                        epochs=10, steps_per_epoch=8,
                        validation_data=([X_val[:, :, :12], X_val[:, :, 12:], X_features], y_val))

    model.save("models/trained_model")
    model.save("models/trained_model_h5.h5")


if __name__ == "__main__":
    main()
