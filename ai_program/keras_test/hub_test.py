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
#二、加载和预处理数据

#2.2 Numpy数据(先下载mnist数据)
'''

DATA_URL = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz'

path = tf.keras.utils.get_file('mnist.npz', DATA_URL)  #从给出的url地址下载数据，查看数据位置：locate mnist.npz #/home/zhangzhipeng/.keras/datasets
with np.load(path) as data:
    train_examples = data['x_train']
    train_labels = data['y_train']
    test_examples = data['x_test']
    test_labels = data['y_test']

print (train_examples.shape)

#将数据切片打包
train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_labels)) #他的所用是切分传入的 Tensor 的第一个维度，生成相应的 dataset 。
test_dataset = tf.data.Dataset.from_tensor_slices((test_examples, test_labels))




#打乱顺序
BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 100

train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE) 
test_dataset = test_dataset.batch(BATCH_SIZE)


#对数据进行训练
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.RMSprop(),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

model.fit(train_dataset, epochs=10)

model.evaluate(test_dataset)

'''



#三、Estimator

#3.1 预创建的Estimator

#数据集的标签，鸢尾花有4个特征(2个长度，2个宽度)，根据这个4个特征可以分为3个species(setosa,versicolor,virginica)或者是(0,1,2)
CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']


#获得鸢尾花数据
train_path = tf.keras.utils.get_file(
    "iris_training.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv")
test_path = tf.keras.utils.get_file(
    "iris_test.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv")

train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)

#print (train.head()) #现在的数据是5列，包括特征以及结果

#移除标签--标准答案。
train_y = train.pop('Species')
test_y = test.pop('Species')

# 标签列现已从数据中删除
print (train.head())


#创建输入函数，相当于将数据预处理
def input_fn(features, labels, training=True, batch_size=256):
    """An input function for training or evaluating"""
    # 将输入转换为数据集。
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # 如果在训练模式下混淆并重复数据。
    if training:
        dataset = dataset.shuffle(1000).repeat()
    
    return dataset.batch(batch_size)

# 特征列描述了如何使用输入。
my_feature_columns = []
for key in train.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))



# 构建一个拥有两个隐层，隐藏节点分别为 30 和 10 的深度神经网络。
classifier = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns,
    # 隐层所含结点数量分别为 30 和 10.
    hidden_units=[30, 10],
    # 模型必须从三个类别中做出选择。
    n_classes=3)


# 训练模型。
classifier.train(
    input_fn=lambda: input_fn(train, train_y, training=True),
    steps=5000)







































































