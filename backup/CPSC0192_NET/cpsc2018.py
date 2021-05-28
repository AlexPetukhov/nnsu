import random
import os
import argparse
import csv
import glob



#添加需要导入的包
import numpy as np
import os
import scipy.io as sio
import tensorflow as tf

#导入模型
from Net import Net

'''
cspc2018_challenge score
Written by:  Xingyao Wang, Feifei Liu, Chengyu Liu
             School of Instrument Science and Engineering
             Southeast University, China
             chengyu@seu.edu.cn
'''

'''
Save prdiction answers to answers.csv in local path, the first column is recording name and the second
column is prediction label, for example:
Recoding    Result
B0001       1
.           .
.           .
.           .
'''

#modeldir = "checkpoints-cnn/4/har.ckpt"

#获取ECG文件
def get_file(record_base_path):
    files = []
    for filename in os.listdir(record_base_path):
        if filename.endswith('.mat'):
            path = os.path.join(record_base_path, filename)
            files.append(path)
    return files



#获取ECG记录文件12导联数据
def get_data(path):
    min_batch = []
    data_12 = sio.loadmat(path)['ECG'][0][0][2]
    filename = []
    for i in range(0,data_12.shape[1],3000):
        jieduan = data_12[:,i:i+3000]
        if jieduan.shape[1] <3000:
            jieduan = data_12[:,-3000:]
        min_batch.append(jieduan[[1,2,6,7,8,9,10,11],:].T)
        filename.append(path.split('/')[-1][:-4])

    return np.array(min_batch).reshape(-1,3000,8),list(set(filename))[0]


#对ECG导联数据做预测
def prediction(datafile):
    tf.reset_default_graph()
    graph = tf.Graph()
    net = Net()

    # 提取变量
    saver = tf.train.Saver()
    sess = tf.Session()
    saver.restore(sess, "checkpoints-cnn/4/har.ckpt")

    prediction_dict = {}

    for filemat in datafile:
        data_mat_pianduan,datafilename = get_data(filemat)

        feed = {net.inputs : data_mat_pianduan, net.keep_prob : 1.0, net.tf_is_train: False}
        prediction2 = sess.run(net.prediction,feed_dict = feed)+1

        if len(set(prediction2)) > 1:

            prediction_ = min(list(set(prediction2)-set([1])))
        else:
            prediction_ = min(list(set(prediction2)))
        mat_file = filemat.split('/')[-1][:-4]
        prediction_dict[mat_file] = prediction_
    print('prediction finished!')

    return prediction_dict

def cpsc2018(record_base_path):
    # ecg = scipy.io.loadmat(record_path)
    ###########################INFERENCE PART################################


    ## Please process the ecg data, and output the classification result.
    ## result should be an integer number in [1, 9].


    ##此处做预测
    datafile = get_file(record_base_path)
    prediction_dict = prediction(datafile)

    with open('answers.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        # column name
        writer.writerow(['Recording', 'Result'])
        for mat_item in os.listdir(record_base_path):
            if mat_item.endswith('.mat') and (not mat_item.startswith('._')):

                #此处做分类对应
                file = mat_item.split('.')[0]
                result = prediction_dict[file]


                # result = random.randint(1, 9)
                ## If the classification result is an invalid number, the result will be determined as normal(1).
                if result > 9 or result < 1 or not(str(result).isdigit()):
                    result = 1
                record_name = mat_item.rstrip('.mat')
                answer = [record_name, result]
                # write result
                writer.writerow(answer)

        csvfile.close()
    ###########################INFERENCE PART################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p',
                        '--recording_path',
                        help='path saving test record file')

    args = parser.parse_args()

    result = cpsc2018(record_base_path=args.recording_path)
