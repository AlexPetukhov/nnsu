import sys

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

sys.path.append('/Users/coubex/nnsu/')
sys.path.append('/home/coubex/nnsu/')

import keras
from keras.losses import SparseCategoricalCrossentropy
import keras.layers as L

from lib.lib import *

filter_means = np.array([32, 64, 128, 256, 512, 256, 128, 64])


def get_conv_model(filter_means, input_layer):
    for i in range(len(filter_means)):
        if i == 0:
            conv = L.Conv1D(filters=filter_means[i], kernel_size=16, padding='same', strides=5, activation='relu')(
                input_layer)
        else:
            conv = L.Conv1D(filters=filter_means[i], kernel_size=16, padding='same', strides=5, activation='relu')(drop)
        drop = L.Dropout(dropout_rate)(conv)
        conv = L.Conv1D(filters=filter_means[i], kernel_size=9, padding='same', strides=5)(drop)
        batch = L.BatchNormalization()(conv)
        relu = L.ReLU()(batch)
        drop = L.Dropout(dropout_rate)(relu)
    return L.Flatten()(drop)


def model_many_diseases2():
    input_layer = L.Input(shape=(3000, 8))

    model_12 = get_conv_model(np.array([32, 64, 32]), input_layer)

    dense_out1 = L.Dense(64, activation='relu')(model_12)
    dense_out2 = L.Dense(9)(dense_out1)

    model = keras.Model(input_layer, [dense_out2])
    return model


def main():
    # 6 sec, no delineation, no features
    # using [[1,2,6,7,8,9,10,11],:] leads
    data_folder_path = '../DATA'
    train_folder_name = 'training_set'
    x_train, y_train = get_train_data(data_folder_path, train_folder_name, need_leads=need_leads)

    X_val = np.load(os.path.join(data_folder_path, '6sec/X_6sec.npy'))
    y_val = np.load(os.path.join(data_folder_path, '6sec/y_6sec.npy'))
    assert (len(X_val) == len(y_val))
    print('len of train data', len(x_train))

    global dropout_rate
    dropout_rate = 0.25

    model = model_many_diseases2()
    model.compile(loss=SparseCategoricalCrossentropy(from_logits=True), optimizer='adam', metrics=['accuracy'])

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto', restore_best_weights=True,
                      min_delta=0.01),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, epsilon=1e-4, mode='min')
    ]
    model.fit(generator_st(x_train=x_train, y_train=y_train, batch_size=128),
              epochs=120, steps_per_epoch=120, callbacks=callbacks,
              validation_data=(X_val[:, :, need_leads], y_val))

    model.save("models/tmp/model.h5")


if __name__ == "__main__":
    main()
