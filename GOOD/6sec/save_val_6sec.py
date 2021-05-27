import os
import numpy as np
import scipy.io as sio
import pandas as pd


def load_data(record_path):
    data = sio.loadmat(record_path)
    signal = data['ECG']['data'][0][0]

    signal = signal.T
    signal_list = []

    for i in range(0, signal.shape[0] - 1500, 3000):
        signal_slice = signal[i:i + 3000, :]
        if signal_slice.shape[0] < 3000:
            signal_slice = signal[-3000:, :]

        signal_list.append(signal_slice)
    return signal_list


def main():
    data_and_delin_path = '../DATA'
    data_path = os.path.join(data_and_delin_path, "validation_set")
    data_list = os.listdir(data_path)
    if 'REFERENCE.csv' in data_list:
        data_list.remove('REFERENCE.csv')
    if '.DS_Store' in data_list:
        data_list.remove('.DS_Store')
    data_list.sort()

    reference_dict = {}
    reference_path = os.path.join(data_path, 'REFERENCE.csv')
    reader = pd.read_csv(reference_path)
    for i, row in reader.iterrows():
        reference_dict[row['Recording']] = int(row['First_label'])

    dataset = []
    for data in data_list:
        cur_data_path = os.path.join(data_path, data)
        signal_list = load_data(cur_data_path)
        name = data.split('.')[0]
        reference_value = reference_dict[name] - 1
        for cur_signal in signal_list:
            dataset.append([name, reference_value, cur_signal])

    X = np.array([x[2] for x in dataset])
    y = np.array([x[1] for x in dataset])
    name = np.array([x[0] for x in dataset])

    np.save('X_6sec', X)
    np.save('y_6sec', y)
    np.save('name_6sec', name)


if __name__ == "__main__":
    main()
