#!/usr/bin/python
# -*- coding:UTF-8 -*-

import numpy as np
import os

import tensorflow as tf

import tensorflow_hub as hub
import tensorflow_datasets as tfds
import pandas as pd
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #忽略弹出的警告。
print(tf.version.VERSION)                #查看版本


                                                                                                  #<1> tf.data.Dataset
#####################################################################################################################

#1. 切片  还可以直接读入txt文档。

'''
dataset = tf.data.Dataset.from_tensor_slices(([1, 2, 3],[4,5,6])) #他的所用是切分传入的 Tensor 的第一个维度，生成相应的 dataset
for element in dataset:
    print(element)

#1.1 对切片的数据进行简单的运算，以及shuffle和batch
#dataset = tf.data.Dataset.from_tensor_slices(([1, 2, 3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9]))
dataset = tf.data.Dataset.from_tensor_slices([1,2,3,4])


#1.2 batch函数
test = dataset.batch(3) #单独的batch只是按顺序取出数据,drop_remainder=False默认是不丢弃最后剩余的(比如11个每次取出3个，剩余2个，可以设置为True，将其丢弃)
for element in test:
    print(element)

#1.3 shuffle和repeat函数
test = dataset.shuffle(3).repeat(2) #单独的shuffle,先取前3个，然后随机取出1个，在补充第4个，再随机取出。 重复2次，是单独重复，相当于dataset.shuffle(3)运行了2次。
for element in test:
    print(element)
test = dataset.repeat(2).shuffle(3) #数据先重复2次，相当与现在的数据是1,2,3,4,1,2,3,4 然后在运行shuffle
for element in test:
    print(element)

#1.4 map函数
#dataset = dataset.map(lambda x,y: x*1)  #lambda只能写一个表达式
for element in dataset:
    print(element)


#1.5 打乱顺序
BATCH_SIZE = 2
SHUFFLE_BUFFER_SIZE = 5

train_dataset = dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE) #首先选择前SHUFFLE_BUFFER_SIZE个数据到队列中，然后随机选出BATCH_SIZE个数据，再补充BATCH_SIZE个数据。

for element in train_dataset:
    print(element)
'''

#3.1简单的写入
import csv

'''
f = open('test.csv','w',encoding='utf-8')
csv_writer = csv.writer(f)
#写入表头
csv_writer.writerow (['a','b','c'])
#写入实际数据
csv_writer.writerow(['SC',1,23])
f.close()                     
                     
                     
'''
#3.2读取第一行有标题的csv格式
import pandas as pd

cata_csv = 'test.csv' 
csv_co = ['00','11','22']
train = pd.read_csv(cata_csv, names=csv_co, header=0) #将第一行作为索引，并且给索引起名字
dict_test  = dict(train) #将读取的csv数据直接整成字典形式，键是第一行的值，对应的值是每一列的数值。
print (dict_test['00'][2])

'''
catalog  = pd.read_csv(cata_csv)
#直接读取所有值，不用担心数据太大，会用省略号代替
#print (catalog) 

#按列读取数据，只显示每一列的第一行
for column in catalog:
    print (column) #显示标题的名称
#
print (catalog['a']) #显示标题为is的那一列的数据的第一个数据

test = dict()
'''










