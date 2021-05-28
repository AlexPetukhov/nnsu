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
    input_layer = L.Input(shape=(3000, 12))

    model_12 = get_conv_model(filter_means[:2], input_layer)

    dense_out1 = L.Dense(512, activation='relu')(model_12)
    dense_out2 = L.Dense(9)(dense_out1)

    model = keras.Model(input_layer, [dense_out2])
    return model


def generator_st(x_train, y_train, batch_size):
    # Create empty arrays to contain batch of features and label
    batch_features_1 = np.zeros((batch_size, 3000, 12))
    batch_features_3 = np.zeros((batch_size, 60))
    batch_labels = np.zeros(batch_size, dtype=int)

    while True:
        for i in range(batch_size):
            # choose random index in features
            index = np.random.choice(len(x_train))
            x = x_train[index][:, :12]
            shift = np.random.randint(0, len(x) - 3000 + 1, 1)[0]

            batch_features_1[i] = x[shift: shift + 3000]
            batch_features_3[i, :12] = batch_features_1[i].max(axis=0)
            batch_features_3[i, 12:24] = batch_features_1[i].min(axis=0)
            batch_features_3[i, 24:36] = batch_features_1[i].mean(axis=0)
            batch_features_3[i, 36:48] = batch_features_1[i].std(axis=0)
            batch_features_3[i, 48:] = batch_features_1[i].max(axis=0) + batch_features_1[i].min(axis=0)
            batch_labels[i] = y_train[index]
        yield [batch_features_1, batch_features_3], batch_labels


def load_data(record_path):
    data = sio.loadmat(record_path)
    signal = data['ECG']['data'][0][0]
    return signal.T


def get_train_data(data_and_delin_path, mat_folder):
    data_path = os.path.join(data_and_delin_path, mat_folder)
    data_list = os.listdir(data_path)
    if 'REFERENCE.csv' in data_list:
        data_list.remove('REFERENCE.csv')
    if '.DS_Store' in data_list:
        data_list.remove('.DS_Store')
    data_list.sort()

    x_train = []
    y_train = []
    reference_dict = {}
    reference_path = os.path.join(data_path, 'REFERENCE.csv')
    reader = pd.read_csv(reference_path)
    for i, row in reader.iterrows():
        reference_dict[row['Recording']] = int(row['First_label'])

    for data in data_list:
        cur_data_path = os.path.join(data_path, data)
        signal = load_data(cur_data_path)
        x_train.append(signal)
        data_name = data.split('.')[0]
        y_train.append(reference_dict[data_name] - 1)
    return x_train, y_train


def main():
    data_and_delin_path = '../../DATA'
    mat_folder = 'training_set'
    x_train, y_train = get_train_data(data_and_delin_path, mat_folder)

    X_val = np.load(os.path.join(data_and_delin_path, '6sec/X_6sec.npy'))
    y_val = np.load(os.path.join(data_and_delin_path, '6sec/y_6sec.npy'))
    X_features = np.array([get_features_from_signal(x) for x in X_val[:, :, :12]])
    print('len(x_train)', len(x_train), len(y_train))


    model = model_many_diseases2()
    model.compile(loss=SparseCategoricalCrossentropy(from_logits=True), optimizer='adam', metrics=['accuracy'])
    model.summary()
    return
    model.fit(generator_st(x_train=x_train, y_train=y_train, batch_size=128),
              epochs=60, steps_per_epoch=60, validation_data=([X_val[:, :, :12], X_features], y_val))

    model.save("models/trained_model")
    model.save("models/trained_model_h5.h5")


if __name__ == "__main__":
    main()
