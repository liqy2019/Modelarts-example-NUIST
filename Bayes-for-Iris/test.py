from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D #空间三维画图
import os
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore")

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

def get_iris():
    dataset = pd.read_csv(local_data_path+"iris.data",names=["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm","Species"]) #读取data文件
    dataset.Species[dataset['Species']=='Iris-setosa']=0
    dataset.Species[dataset['Species']=='Iris-versicolor']=1
    dataset.Species[dataset['Species']=='Iris-virginica']=2
    data = np.array(dataset.loc[0:149, ['SepalLengthCm', 'SepalWidthCm','PetalLengthCm','PetalWidthCm']])       #打印行中特定列

    target = np.array(dataset.loc[0:149, ['Species']]).squeeze()
    
    return data,target

data,target=get_iris()
print(target)#打印种类的值


#划分测试集和训练集
def split_iris():
    from sklearn import model_selection
    data,target=get_iris()
    all_inputs=data
    all_classes=target
    train_inputs,test_inputs,train_classes,test_classes=model_selection.train_test_split(all_inputs,all_classes,train_size=0.75,random_state=1)#random_state制定随机数的种子
    return train_inputs,test_inputs,train_classes,test_classes

train_data,test_data,train_classes,test_classes=split_iris()
#归一化
def guiyi(data):
    import numpy as np   
    a1=np.mean(data,axis=0)#求均值
    stdr=np.std(data,ddof=1,axis=0)#ddof无偏样本标准差
    y=(data-a1)/stdr
    return y

#最小错误率贝叶斯分类
def bayes(test_da,test_cla):
    import numpy as np    
    data,target=get_iris()
    data=guiyi(data)
    test_dat=guiyi(test_da)
    r=[50,50,50]#每个类别的样本个数
    r1=3#类别为3
    #loss=np.ones(r1)-np.diag(np.ones(r1))#风险率
    
    clas=[]
    for k in range(np.shape(test_da)[0]):
        temp=0
        ii=[]
        
        for i in range(r1):
            y=data[temp:r[i]+temp]
            temp=temp+r[i]
            y_cov=np.cov(y.T)#求协方差
            a=np.mat(y_cov)
            y_inv=np.linalg.inv(a)           
            y_det=np.linalg.det(y_cov)#求行列式
            y_mean=np.mean(y,axis=0)
            p=r[i]/np.sum(r)
            z=test_dat[k,:]-y_mean
            h=-z*y_inv*z.reshape(4,1)/2+np.log(p)-np.log(np.abs(y_det))/2;        
            ii.append(h)
        #print('\n测试集第{0}个样本在三个判别函数的值为:\n{1}'.format(k+1,ii))
        a,re=max(ii),ii.index(max(ii))
        #print('概率最大为{0}，类别为{1}'.format(a,re))
        clas.append(re)
    print('\n测试结果\n{0}'.format(clas))
    print('正确结果\n{0}'.format(test_cla))
    #计算准确率
    count=0
    for i in range(len(clas)):       
        if clas[i]==test_cla[i]:
            count+=1
    acc=count/len(clas)*100
    print('测试的样本个数为{0},测试正确的个数为{1}'.format(len(clas),count))
    print('准确率为%.2f%%'%acc)
train_data,test_data,train_classes,test_classes=split_iris()#划分测试集和训练集
print("最小错误率贝叶斯分类")
bayes(train_data,train_classes)
bayes(test_data,test_classes)
#最小风险率贝叶斯分类    
def bayesmi(test_da,test_cla):
    import numpy as np    
    data,target=get_iris()
    data=guiyi(data)
    test_dat=guiyi(test_da)
    r=[50,50,50]#每个类别的样本个数
    r1=3#类别为3
    loss=np.ones((r1,r1))-np.diag(np.ones((r1)))#风险率
    
    clas=[]
    
    for k in range(np.shape(test_dat)[0]):
        temp=0
        ii=[]
        risk=[]
        
        for i in range(r1):
            y=data[temp:r[i]+temp]
            temp=temp+r[i]
            y_cov=np.cov(y.T)#求协方差
            a=np.mat(y_cov)
            y_inv=np.linalg.inv(a)           
            y_det=np.linalg.det(y_cov)#求行列式
            y_mean=np.mean(y,axis=0)
            p=r[i]/np.sum(r)
            z=test_dat[k,:]-y_mean
            h=-z*y_inv*z.reshape(4,1)/2+np.log(p)-np.log(np.abs(y_det))/2;        
            ii.append(h)
        for i in range(r1):
            ris=np.dot(loss[i,:],np.array(ii).reshape(3,1))
            risk.append(ris)
        #print('\n测试集第{0}个样本在三个判别函数的值为:\n{1}'.format(k+1,risk))
        a,re=min(risk),risk.index(min(risk))
        #print('风险概率最小为{0}，类别为{1}'.format(a,re))
        clas.append(re)
    print('\n测试结果\n{0}'.format(clas))
    print('正确结果\n{0}'.format(test_cla))
    #计算准确率
    count=0
    for i in range(len(clas)):       
        if clas[i]==test_cla[i]:
            count+=1
    acc=count/len(clas)*100
    print('测试的样本个数为{0},测试正确的个数为{1}'.format(len(clas),count))
    print('准确率为%.2f%%'%acc)
print("最小风险率贝叶斯分类")
bayesmi(test_data,test_classes)
bayesmi(train_data,train_classes)

