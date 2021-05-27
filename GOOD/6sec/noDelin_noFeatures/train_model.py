import sys
sys.path.append('/Users/coubex/nnsu/')
sys.path.append('/home/coubex/nnsu/')

import keras
from keras.losses import SparseCategoricalCrossentropy
import keras.layers as L

from lib.lib import *

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
    input_layer = L.Input(shape=(3000, 8))

    model_12 = get_conv_model(filter_means[:2], input_layer)

    dense_out1 = L.Dense(512, activation='relu')(model_12)
    dense_out2 = L.Dense(9)(dense_out1)

    model = keras.Model(input_layer, [dense_out2])
    return model


def generator_st(x_train, y_train, batch_size):
    # Create empty arrays to contain batch of features and label
    batch_features_1 = np.zeros((batch_size, 3000, 8))
    batch_labels = np.zeros(batch_size, dtype=int)

    while True:
        for i in range(batch_size):
            # choose random index in features
            index = np.random.choice(len(x_train))
            x = x_train[index][:, :8]
            shift = np.random.randint(0, len(x) - 3000 + 1, 1)[0]

            batch_features_1[i] = x[shift: shift + 3000]
            batch_labels[i] = y_train[index]
        yield batch_features_1, batch_labels


def get_train_data(data_folder_path, train_folder_name, need_leads):
    train_folder_path = os.path.join(data_folder_path, train_folder_name)
    data_list = get_sorted_data_list(train_folder_path)

    x_train = []
    y_train = []
    reference_path = os.path.join(train_folder_path, 'REFERENCE.csv')
    reference_dict = get_reference_dict(path=reference_path, one_value_only=True)  # 1 value only

    for data in data_list:
        cur_data_path = os.path.join(train_folder_path, data)
        signal = get_signal(cur_data_path)
        signal = signal[:, need_leads]
        data_name = data.split('.')[0]

        x_train.append(signal)
        y_train.append(reference_dict[data_name] - 1)

    return x_train, y_train


def main():
    sys.path.append('~/nnsu')
    # 6 sec, no delineation, no features
    # using [[1,2,6,7,8,9,10,11],:] leads
    data_folder_path = '../DATA'
    train_folder_name = 'training_set'
    x_train, y_train = get_train_data(data_folder_path, train_folder_name, need_leads=need_leads)

    X_val = np.load(os.path.join(data_folder_path, '6sec/X_6sec.npy'))
    y_val = np.load(os.path.join(data_folder_path, '6sec/y_6sec.npy'))
    assert(len(X_val) == len(y_val))
    print('len of train data', len(x_train))

    model = model_many_diseases2()
    model.compile(loss=SparseCategoricalCrossentropy(from_logits=True), optimizer='adam', metrics=['accuracy'])

    model.fit(generator_st(x_train=x_train, y_train=y_train, batch_size=128),
              epochs=60, steps_per_epoch=60, validation_data=(X_val[:, :, need_leads], y_val))

    model.save("models/model_6sec_noDelin_noFeatures")
    model.save("models/model_6sec_noDelin_noFeatures.h5")
    model.save("models/tmp/model.h5")


if __name__ == "__main__":
    main()
