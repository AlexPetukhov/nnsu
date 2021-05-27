import sys

sys.path.append('/Users/coubex/nnsu/')
sys.path.append('/home/coubex/nnsu/')

from keras.models import load_model
from lib.lib import *


# using [[1,2,6,7,8,9,10,11],:] leads


def load_and_predict_data_for_one_slice(data_path, start_t, end_t):
    signal = get_signal_slice(data_path, start_t, end_t)
    signal = signal[:, need_leads]
    result = model.predict(np.expand_dims(signal, 0))
    return result.tolist()[0]


def load_and_predict_data_sample(data_path):
    data = sio.loadmat(data_path)
    signal = data['ECG']['data'][0][0]
    max_len = signal.shape[1]

    answer_list = [0] * 9
    for start_t in range(0, max_len - 3000, 500):
        res = load_and_predict_data_for_one_slice(data_path, start_t, start_t + 3000)
        answer_list = [x + y for x, y in zip(answer_list, res)]
    answer = np.argmax(answer_list) + 1
    return answer


def write_results(results):
    print("writing results...")
    data_reference_dict = get_reference_dict(validation_path + "/REFERENCE.csv", one_value_only=False)
    with open("results.csv", 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['№', 'Recording', 'Reference', 'Prediction', 'IsCorrect'])
        correct_predictions = sum([value in data_reference_dict[key] for key, value in results.items()])
        for idx, (name, value) in enumerate(results.items()):
            if idx % (len(results) / 10) == 0:
                print(name)

            reference = ", ".join(str(x) for x in data_reference_dict[name])
            writer.writerow([idx + 1, name, reference, value, value in data_reference_dict[name]])
        writer.writerow([len(results), '', '', '', correct_predictions / len(results)])
        csvfile.close()


def main():
    # using [[1,2,6,7,8,9,10,11],:] leads
    # 6 sec, no delineation, no features
    global model, validation_path
    model = load_model("models/model_6sec_noDelin_noFeatures.h5")
    validation_path = '../DATA/validation_set'

    data_list = get_sorted_data_list(validation_path)
    print('data_list size:', len(data_list))

    model.summary()

    results = {}
    for idx, data in enumerate(data_list):
        name = data.split('.')[0]
        if idx % (len(data_list) / 10) == 0:
            print(name)

        data_path = os.path.join(validation_path, data)
        result = load_and_predict_data_sample(data_path)
        results[name] = result

    write_results(results)


if __name__ == "__main__":
    main()
