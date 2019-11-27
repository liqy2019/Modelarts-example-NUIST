import os, random
import numpy as np

from sklearn import svm
import pandas as pd  #导入pandas包
import tensorflow as tf
from moxing.framework import file
'''
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--data_url",  default='/cache/data', help="dataset directory.")
parser.add_argument("--train_url", default='/cache/output', help="saved model directory.")
args = parser.parse_args()

data_url = args.data_url
train_url = args.train_url
'''

tf.flags.DEFINE_string('data_url', '/cache/data', 'dataset directory.')
tf.flags.DEFINE_string('train_url', '/cache/output', 'saved model directory.')


FLAGS = tf.flags.FLAGS

data_url = FLAGS.data_url
train_url = FLAGS.train_url

'''
#from absl import flags
#flags.DEFINE_string('data_url', '/cache/data', 'dataset directory.')
#flags.DEFINE_string('train_url', '/cache/output', 'saved model directory.')
#FLAGS = flags.FLAGS
#data_url = FLAGS.data_url
#train_url = FLAGS.train_url
'''

'''
data_url = '/cache/data'
train_url = '/cache/output'
'''
'''
data_url = '/lqy-3/Iris-Date/'
train_url = '/lqy-3/output-test/V0016/'
'''

# local path
local_data_path = '/cache/data/'
local_output_path = '/cache/output/'
model_output_path = os.path.join(local_output_path, "model")

if not os.path.exists(local_data_path):
    os.makedirs(local_data_path)

if not os.path.exists(local_output_path):
    os.makedirs(local_output_path)

# copy data to local
file.copy_parallel(data_url, local_data_path)

data = pd.read_csv(local_data_path+"iris.data",names=["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm","Species"])   #读取csv文件
print(data)
print(data.columns)                         #返回全部列名
print(data.shape)                           #f返回csv文件形状
print(data.loc[1:2])                        #打印第1到2行
x = np.array(data.loc[0:149, ['SepalLengthCm', 'SepalWidthCm','PetalLengthCm','PetalWidthCm']])       #打印行中特定列

y = np.array(data.loc[0:149, ['Species']]).squeeze()
print(x.shape)
num = x.shape[0]  # 样本总数
print(num)
ratio = 7 / 3  # 划分比例，训练集数目:测试集数目
num_test = int(num / (1 + ratio))  # 测试集样本数目
num_train = num - num_test  # 训练集样本数目
index = np.arange(num)  # 产生样本标号
np.random.shuffle(index)  # 洗牌
x_test = x[index[:num_test], :]  # 取出洗牌后前 num_test 作为测试集
y_test = y[index[:num_test]]
x_train = x[index[num_test:], :]  # 剩余作为训练集
y_train = y[index[num_test:]]

clf_linear = svm.SVC(decision_function_shape="ovo", kernel="linear")
clf_rbf = svm.SVC(decision_function_shape="ovo", kernel="rbf")
clf_linear.fit(x_train, y_train)
clf_rbf.fit(x_train, y_train)

y_test_pre_linear = clf_linear.predict(x_test)
y_test_pre_rbf = clf_rbf.predict(x_test)

# 计算分类准确率
acc_linear = sum(y_test_pre_linear == y_test) / num_test
print('linear kernel: The accuracy is', acc_linear)
acc_rbf = sum(y_test_pre_rbf == y_test) / num_test
print('rbf kernel: The accuracy is', acc_rbf)