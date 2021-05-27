
import csv
import os
import tensorflow as tf
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio


model = load_model("../../MODELS/viktor.h5")
data_folder_path = '../../DATA/validation_set'
delin_folder_path = '../../DATA/delineation_leads_val'


def load_data(record_path, delineation_path, start_t, end_t, interval: int):
    data = sio.loadmat(record_path)
    signal = data['ECG']['data'][0][0]
    signal = signal[:, start_t:end_t:interval]

    delineation = np.load(delineation_path)
    delineation = delineation[:, start_t:end_t:interval]

    features = np.empty(60)
    features[:12] = signal.max(axis=1)
    features[12:24] = signal.min(axis=1)
    features[24:36] = signal.mean(axis=1)
    features[36:48] = signal.std(axis=1)
    features[48:] = signal.max(axis=1) + signal.min(axis=1)

    return np.expand_dims(signal.T, 0), np.expand_dims(delineation.T, 0), np.expand_dims(features, 0)


def get_data_reference_dict(reference_path) -> dict:
    reference_dict = {}
    with open(reference_path, "r") as f_obj:
        reader = csv.reader(f_obj)
        for row in reader:
            name = row[0]
            if name == 'Recording':
                continue
            del row[0]
            reference_dict[name] = [int(x) for x in row if x]
            reference_dict[name].sort()
    return reference_dict


def get_data_and_delin_list(data_path, delin_path):
    data_list = os.listdir(data_path)
    delin_list = os.listdir(delin_path)
    data_list.sort()
    delin_list.sort()
    data_reference_dict = get_data_reference_dict("validation_set/REFERENCE.csv")
    if 'REFERENCE.csv' in data_list:
        data_list.remove('REFERENCE.csv')
    if '.DS_Store' in data_list:
        data_list.remove('.DS_Store')
    if 'REFERENCE.csv' in delin_list:
        delin_list.remove('REFERENCE.csv')
    if '.DS_Store' in delin_list:
        delin_list.remove('.DS_Store')
    delin_list = [x for x in delin_list if str(f'A{x.split(".")[0]}.mat') in data_list]
    return data_list, delin_list, data_reference_dict

def load_and_predict_data_for_one_slice(data_path, delin_path, start_t, end_t, interval):
    signal, delineation, features = load_data(data_path, delin_path, start_t, end_t, interval)
    prediction = model.predict([signal, delineation, features])
    result = norm_predict(prediction)
    return prediction, result

def load_and_predict_data(data_path, delin_path):
    prediction = []
    result = []
    data = sio.loadmat(data_path)
    signal = data['ECG']['data'][0][0]
    max_len = signal.shape[1]

    interval = 1
    for start_t in range(0, max_len - 4500, 500):
        pred, res = load_and_predict_data_for_one_slice(data_path, delin_path, start_t, start_t + 4500, interval)
        prediction.append(pred)
        result.append(res)

    if max_len >= 9000:
        interval = 2
        for start_t in range(0, max_len - 9000, 500):
            pred, res = load_and_predict_data_for_one_slice(data_path, delin_path, start_t, start_t + 9000, interval)
            prediction.append(pred)
            result.append(res)
    return prediction, set(result)


def main():
    data_list, delin_list, data_reference_dict = get_data_and_delin_list(data_folder_path, delin_folder_path)
    print('good data size:', len(data_list))
    results = {}
    for data, delin in zip(data_list, delin_list):
        data_path = os.path.join(data_folder_path, data)
        delin_path = os.path.join(delin_folder_path, delin)
        prediction, result = load_and_predict_data(data_path, delin_path)
        results[data] = result

    with open("results.csv", 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['№', 'Recording', 'Reference', 'Prediction', 'Correct'])
        correct_predictions = 0
        for idx, (key, value) in enumerate(results.items()):
            name = key.split('.')[0]
            reference = ", ".join(str(x) for x in data_reference_dict[name])
            # if value in data_reference_dict[name]:
            #     correct_predictions = correct_predictions + 1

            if len(list(set(value) & set(data_reference_dict[name]))) > 0:
                correct_predictions = correct_predictions + 1

            value_str = ", ".join(str(x) for x in value)
            writer.writerow(
                [idx + 1, name, reference, value_str, len(list(set(value) & set(data_reference_dict[name]))) > 0])
        writer.writerow([len(results), '', '', '', correct_predictions / len(results)])
        csvfile.close()


if __name__ == "__main__":
    main()
