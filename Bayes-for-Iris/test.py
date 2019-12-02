import pandas as pd
import numpy as np
import scipy.stats as stats
import os

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

dataset = pd.read_csv(local_data_path+"iris.data",names=["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm","Species"])   #读取data文件

class Bayes_Test():

    # 提取样本特征集和类别集 划分训练/测试集
    def split_out_dataset(self, dataset):
        array = dataset.values  # 将数据库转换成数组形式
        X = array[:, 0:4].astype(float)  # 取特征数值列
        Y = array[:, 4]  # 取类别列
        return X,  Y

    """第一步：划分样本集"""
    # 提取样本 特征
    def split_out_attributes(self, X, Y):
        # 提取 每个类别的不同特征
        # c1 第一类的特征数组
        c1_1 = c1_2 = c1_3 = c1_4 = []
        c2_1 = c2_2 = c2_3 = c2_4 = []
        c3_1 = c3_2 = c3_3 = c3_4 = []
        for i in range(len(Y)):
            if (Y[i] == 'Iris-setosa'):
                c1_1.append(X[i, 0])
                c1_2.append(X[i, 1])
                c1_3.append(X[i, 2])
                c1_4.append(X[i, 3])
            elif (Y[i] == 'Iris-versicolor'):
                # c2 第二类的特征数组
                c2_1.append(X[i, 0])
                c2_2.append(X[i, 1])
                c2_3.append(X[i, 2])
                c2_4.append(X[i, 3])
            elif (Y[i] == 'Iris-virginica'):
                # c3 第三类的特征数组
                c3_1.append(X[i, 0])
                c3_2.append(X[i, 1])
                c3_3.append(X[i, 2])
                c3_4.append(X[i, 3])
            else:
                pass

        return [c1_1, c1_2, c1_3, c1_4,
                c2_1, c2_2, c2_3, c2_4,
                c3_1, c3_2, c3_3, c3_4]

    """因为符合多变量正态分布，所以需要(μ，∑)两个样本参数"""

    """第二步：计算样本期望μ和样本方差s"""
    # 计算样本期望
    def cal_mean(self, attributes):

        c1_1, c1_2, c1_3, c1_4, c2_1, c2_2, c2_3, c2_4, c3_1, c3_2, c3_3, c3_4 = attributes
        # 第一类的期望值μ
        e_c1_1 = np.mean(c1_1)
        e_c1_2 = np.mean(c1_2)
        e_c1_3 = np.mean(c1_3)
        e_c1_4 = np.mean(c1_4)
        # 第二类的期望值μ
        e_c2_1 = np.mean(c2_1)
        e_c2_2 = np.mean(c2_2)
        e_c2_3 = np.mean(c2_3)
        e_c2_4 = np.mean(c2_4)
        # 第三类的期望值μ
        e_c3_1 = np.mean(c3_1)
        e_c3_2 = np.mean(c3_2)
        e_c3_3 = np.mean(c3_3)
        e_c3_4 = np.mean(c3_4)
        return [e_c1_1, e_c1_2, e_c1_3, e_c1_4,
                e_c2_1, e_c2_2, e_c2_3, e_c2_4,
                e_c3_1, e_c3_2, e_c3_3, e_c3_4]

    # 计算样本方差

    def cal_var(self, attributes):
        c1_1, c1_2, c1_3, c1_4, c2_1, c2_2, c2_3, c2_4, c3_1, c3_2, c3_3, c3_4 = attributes
        # 第一类的方差var
        var_c1_1 = np.var(c1_1)
        var_c1_2 = np.var(c1_2)
        var_c1_3 = np.var(c1_3)
        var_c1_4 = np.var(c1_4)
        # 第二类的方差s
        var_c2_1 = np.var(c2_1)
        var_c2_2 = np.var(c2_2)
        var_c2_3 = np.var(c2_3)
        var_c2_4 = np.var(c2_4)
        # 第三类的方差s
        var_c3_1 = np.var(c3_1)
        var_c3_2 = np.var(c3_2)
        var_c3_3 = np.var(c3_3)
        var_c3_4 = np.var(c3_4)
        return [var_c1_1, var_c1_2, var_c1_3, var_c1_4,
                var_c2_1, var_c2_2, var_c2_3, var_c2_4,
                var_c3_1, var_c3_2, var_c3_3, var_c3_4]
    # 计算先验概率P(Y=ck)
    def cal_prior_probability(self, Y):
        a = b = c = 0
        for i in Y:
            if (i == 'Iris-setosa'):
                a += 1
            elif (i == 'Iris-versicolor'):
                b += 1
            elif (i == 'Iris-virginica'):
                c += 1
            else:
                pass
        pa = a / len(Y)
        pb = b / len(Y)
        pc = c / len(Y)
        return pa, pb, pc
    # 计算后验概率P(Y=ck|X)=P(X|Y=ck)*P(Y=ck)/∑
    def cal_posteriori_probability(self, X, Y, p, means, vars):
        pa, pb, pc = p
        e_c1_1, e_c1_2, e_c1_3, e_c1_4, e_c2_1, e_c2_2, e_c2_3, e_c2_4, e_c3_1, e_c3_2, e_c3_3, e_c3_4 = means
        var_c1_1, var_c1_2, var_c1_3, var_c1_4, var_c2_1, var_c2_2, var_c2_3, var_c2_4, var_c3_1, var_c3_2, var_c3_3, var_c3_4 = vars

        print('p:', p)
        print('means:', means)
        print('vars:', vars)
        # 分解四维输入向量X=[X1，X2，X3，X4]为4个一维正态分布函数
        X1 = X[:, 0]
        X2 = X[:, 1]
        X3 = X[:, 2]
        X4 = X[:, 3]

        # 分类正确数/分类错误数=>计算正确率
        true_test = 0
        false_test = 0
        # 遍历训练整个输入空间，计算后验概率并判决
        for i in range(len(X1)):
            # 计算后验概率=P(X|Y=C1)P(Y=C1)
            P_1 = stats.norm.pdf(X1[i], e_c1_1, var_c1_1) * stats.norm.pdf(X2[i], e_c1_2, var_c1_2) * stats.norm.pdf(
                X3[i], e_c1_3,
                var_c1_3) * stats.norm.pdf(
                X4[i], e_c1_4, var_c1_4) * pa
            # 计算后验概率=P(X|Y=C2)P(Y=C2)
            P_2 = stats.norm.pdf(X1[i], e_c2_1, var_c2_1) * stats.norm.pdf(X2[i], e_c2_2, var_c2_2) * stats.norm.pdf(
                X3[i], e_c2_3,
                var_c2_3) * stats.norm.pdf(
                X4[i], e_c2_4, var_c2_4) * pb
            # 计算后验概率=P(X|Y=C3)P(Y=C3)
            P_3 = stats.norm.pdf(X1[i], e_c3_1, var_c3_1) * stats.norm.pdf(X2[i], e_c3_2, var_c3_2) * stats.norm.pdf(
                X3[i], e_c3_3,
                var_c3_3) * stats.norm.pdf(
                X4[i], e_c3_4, var_c3_4) * pc

            # 计算判别函数，选取概率最大的类
            max_P = max(P_1, P_2, P_3)
            # 输出分类结果，并检测正确率
            if (max_P == P_1):
                if (Y[i] == 'Iris-setosa'):
                    print('分为第一类，正确')
                    true_test += 1
                else:
                    print('分为第一类，错误')
                    false_test += 1
            elif (max_P == P_2):
                if (Y[i] == 'Iris-versicolor'):
                    print('分为第二类，正确')
                    true_test += 1
                else:
                    print('分为第二类，错误')
                    false_test += 1
            elif (max_P == P_3):
                if (Y[i] == 'Iris-virginica'):
                    print('分为第三类，正确')
                    true_test += 1
                else:
                    print('分为第三类，错误')
                    false_test += 1
            else:
                print('未分类')
                false_test += 1
        # 打印分类正确率
        print('贝叶斯分类器正确率为:', (true_test / (true_test + false_test)))
bayes = Bayes_Test()
# 划分训练集 测试集
X_train, Y_train= bayes.split_out_dataset(dataset)
print('得到的X_train', X_train)
print('得到的Y_train', Y_train)
# 分割属性--训练集
attributes = bayes.split_out_attributes(X_train, Y_train)
print('得到的训练集', attributes)
# 计算期望--训练集
means = bayes.cal_mean(attributes)
print('得到的means', means)
# 计算方差--训练集
vars = bayes.cal_var(attributes)
print('得到的vars', vars)
# 计算先验概率--训练集
prior_p = bayes.cal_prior_probability(Y_train)
# 验证分类准确性--训练集
bayes.cal_posteriori_probability(X_train, Y_train, prior_p, means, vars)


