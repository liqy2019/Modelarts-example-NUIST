# 导入 jieba
import jieba
import os

import tensorflow as tf
from moxing.framework import file

tf.flags.DEFINE_string('data_url', '/cache/data', 'dataset directory.')
tf.flags.DEFINE_string('train_url', '/cache/output', 'saved model directory.')


FLAGS = tf.flags.FLAGS

data_url = FLAGS.data_url
train_url = FLAGS.train_url

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

# 载入词典
jieba.load_userdict(local_data_path+"pku_training.txt")

# 全模式
seg_list = jieba.cut("他来到南京信息工程大学", cut_all=True,HMM=True)  #使用HMM
print("【全模式】：" + "/ ".join(seg_list))

# 精确模式
seg_list = jieba.cut("他来到南京信息工程大学", cut_all=False,HMM=True)   #使用HMM
print("【精确模式】：" + "/ ".join(seg_list))


