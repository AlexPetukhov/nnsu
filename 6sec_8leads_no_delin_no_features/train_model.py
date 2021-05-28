import sys

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

sys.path.append('/Users/coubex/nnsu/')
sys.path.append('/home/coubex/nnsu/')

import keras
from keras.losses import SparseCategoricalCrossentropy
import keras.layers as L

from lib.lib import *

filter_means = np.array([32, 64, 128, 256, 512, 256, 128, 64])


def get_conv_model(filter_means, input_layer, dropout_rate):
    for i in range(len(filter_means)):
        if i == 0:
            conv = L.Conv1D(filters=filter_means[i], kernel_size=9, padding='same', strides=5, activation='relu')(
                input_layer)
        else:
            conv = L.Conv1D(filters=filter_means[i], kernel_size=9, padding='same', strides=5, activation='relu')(relu)
        drop = L.Dropout(dropout_rate)(conv)
        conv = L.Conv1D(filters=filter_means[i], kernel_size=9, padding='same', strides=5)(drop)
        batch = L.BatchNormalization()(conv)
        relu = L.ReLU()(batch)
        drop = L.Dropout(dropout_rate)(relu)
    return L.Flatten()(drop)


def model_many_diseases2(dropout_rate):
    input_layer = L.Input(shape=(3000, 12))

    model_12 = get_conv_model(np.array([32, 32]), input_layer, dropout_rate)

    dense_out1 = L.Dense(512, activation='relu')(model_12)
    dense_out2 = L.Dense(9)(dense_out1)

    model = keras.Model(input_layer, [dense_out2])
    return model


def main():
    # 6 sec, no delineation, no features, 8 leads
    # using [[1,2,6,7,8,9,10,11],:] leads
    data_folder_path = '../DATA'
    train_folder_name = 'training_set'
    x_train, y_train = get_train_data(data_folder_path, train_folder_name, need_leads=need_leads)

    X_val = np.load(os.path.join(data_folder_path, '6sec/X_6sec.npy'))
    y_val = np.load(os.path.join(data_folder_path, '6sec/y_6sec.npy'))
    assert (len(X_val) == len(y_val))
    print('len of train data', len(x_train))
    print('shapes:')
    print('x_train', len(x_train), x_train[0].shape)
    print('y_train', len(y_train))
    print('X_val', X_val.shape)
    print('y_val', y_val.shape)

    model = model_many_diseases2(dropout_rate=0.25)
    model.summary()
    model.compile(loss=SparseCategoricalCrossentropy(from_logits=True), optimizer='adam', metrics=['accuracy'])

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto', restore_best_weights=True,
                      min_delta=0.001),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, min_delta=1e-4, mode='auto')
    ]
    model.fit(generator_st(x_train=x_train, y_train=y_train, batch_size=128, slice_len=3000, need_leads=need_leads),
              epochs=120, steps_per_epoch=120, callbacks=callbacks,
              validation_data=(X_val[:, :, need_leads], y_val))

    model.save("models/tmp/model.h5")


if __name__ == "__main__":
    main()


def train_model():
    main()
