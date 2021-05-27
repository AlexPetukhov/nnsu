import csv
import os
import pandas as pd
import numpy as np
import scipy.io as sio

need_leads = [1, 2, 6, 7, 8, 9, 10, 11]


def get_signal_slice(record_path, start_t, end_t):
    data = sio.loadmat(record_path)
    signal = data['ECG']['data'][0][0]
    signal_slice = signal[:, start_t:end_t]
    return signal_slice.T


def get_signal(record_path):
    data = sio.loadmat(record_path)
    signal = data['ECG']['data'][0][0]
    return signal.T


def get_sorted_data_list(path):
    data_list = os.listdir(path)
    if 'REFERENCE.csv' in data_list:
        data_list.remove('REFERENCE.csv')
    if '.DS_Store' in data_list:
        data_list.remove('.DS_Store')
    data_list.sort()
    return data_list


def get_reference_dict(path: str, one_value_only: bool):
    reference_dict = {}
    if one_value_only:
        reader = pd.read_csv(path)
        for i, row in reader.iterrows():
            reference_dict[row['Recording']] = int(row['First_label'])
    else:
        with open(path, "r") as f_obj:
            reader = csv.reader(f_obj)
            for row in reader:
                name = row[0]
                if name == 'Recording':
                    continue
                del row[0]
                reference_dict[name] = [int(x) for x in row if x]
                reference_dict[name].sort()
    return reference_dict


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
