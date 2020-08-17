#!/usr/bin/python
# -*- coding:UTF-8 -*-

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #忽略弹出的警告。


#1.1 读取csv数据
URL = 'https://storage.googleapis.com/applied-dl/heart.csv'
dataframe = pd.read_csv(URL)  #总长度是303
#print (dataframe.head())
#print (len(dataframe))

#1.1.2使用轮子将数据分成训练和测试数据(竟然是随机的，非常方便)
train, test = train_test_split(dataframe, test_size=0.2)
#print(len(train),len(test))#测试数据占20%

#1.1.3将训练数据分成训练和验证
train, val = train_test_split(train, test_size=0.2)
#print(len(train),len(val))


#1.1.4切字典
#ds = tf.data.Dataset.from_tensor_slices(dict(train))  
#for element in ds:
#    print (element)
#    break


# 一种从 Pandas Dataframe 创建 tf.data 数据集的实用程序方法（utility method）
def df_to_dataset(dataframe, shuffle=True, batch_size=32):
    dataframe = dataframe.copy()
    labels = dataframe.pop('target') #将最后一行即‘target’这一列单独拎出来赋值给labels,同时删除dataframe中的labels
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels)) #切片了，但是所有的数据仍然在ds中
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))                    #打乱顺序
    ds = ds.batch(batch_size)                                          #分组，5个一组。
    return ds


#1.1.5将训练、验证、测试数据切片，每个batch都是5个数据
batch_size = 5 # 小批量大小用于演示
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

#print (dir(train_ds))



'''
#1.1.6查看其中的部分数据
for element in train_ds:
    print (element)
    break


for feature_batch, label_batch in train_ds.take(1):
    print('Every feature:', list(feature_batch.keys()))
    print('A batch of ages:', feature_batch['age'])
    print('A batch of targets:', label_batch )
    
'''
#1.1.7 使用迭代器查看第一个batch中的数据，注意里面是2组数据，一个是完整的，一个是target
example_batch = next(iter(train_ds))[0]
print (example_batch)



# 用于创建一个特征列
# 并转换一批次数据的一个实用程序方法
def demo(feature_column):
    feature_layer = layers.DenseFeatures(feature_column)
    print(feature_layer(example_batch).numpy())


age = feature_column.numeric_column("age")
demo(age)








