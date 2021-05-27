import csv
import os
from keras.models import load_model
import numpy as np
import scipy.io as sio


model = load_model("models/trained_model_h5.h5")


def load_data(record_path, start_t, end_t):
    data = sio.loadmat(record_path)
    signal = data['ECG']['data'][0][0]
    signal = signal[:, start_t:end_t]

    features = np.empty(60)
    features[:12] = signal.max(axis=1)
    features[12:24] = signal.min(axis=1)
    features[24:36] = signal.mean(axis=1)
    features[36:48] = signal.std(axis=1)
    features[48:] = signal.max(axis=1) + signal.min(axis=1)

    return np.expand_dims(signal.T, 0), np.expand_dims(features, 0)


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


def get_data_and_delin_list(data_path):
    data_list = os.listdir(data_path)
    data_list.sort()
    reference_path = os.path.join(data_path, "REFERENCE.csv")
    data_reference_dict = get_data_reference_dict(reference_path)
    if 'REFERENCE.csv' in data_list:
        data_list.remove('REFERENCE.csv')
    if '.DS_Store' in data_list:
        data_list.remove('.DS_Store')
    return data_list, data_reference_dict


def load_and_predict_data_for_one_slice(data_path, start_t, end_t):
    signal, features = load_data(data_path, start_t, end_t)
    prediction = model.predict([signal, features])
    result = prediction.tolist()
    return prediction, result[0]


def load_and_predict_data(data_path):
    prediction = []
    result = []
    data = sio.loadmat(data_path)
    signal = data['ECG']['data'][0][0]
    max_len = signal.shape[1]

    for start_t in range(0, max_len - 3000, 500):
        pred, res = load_and_predict_data_for_one_slice(data_path, start_t, start_t + 3000)
        prediction.append(pred)
        result.append(res)

    answer_list = []
    for i in range(len(result[0])):
        sum = 0
        for j in range(len(result)):
            sum = sum + result[j][i]
        answer_list.append(sum)
    answer = np.argmax(answer_list) + 1
    return prediction, answer, answer_list


def main():
    data_folder_path = '../DATA/validation_set'

    data_list, data_reference_dict = get_data_and_delin_list(data_folder_path)
    print('good data size:', len(data_list))
    results = {}
    for data in data_list:
        data_path = os.path.join(data_folder_path, data)
        prediction, result, result_list = load_and_predict_data(data_path)
        results[data] = result

    with open("results.csv", 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['№', 'Recording', 'Reference', 'Prediction', 'Correct'])
        correct_predictions = 0
        for idx, (key, value) in enumerate(results.items()):
            name = key.split('.')[0]
            reference = ", ".join(str(x) for x in data_reference_dict[name])

            if value in data_reference_dict[name]:
                correct_predictions = correct_predictions + 1

            writer.writerow([idx + 1, name, reference, value, value in data_reference_dict[name]])
        writer.writerow([len(results), '', '', '', correct_predictions / len(results)])
        csvfile.close()


if __name__ == "__main__":
    main()
