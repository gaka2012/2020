#!/usr/bin/python
# -*- coding:UTF-8 -*-


# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
#import matplotlib.pyplot as plt
import cv2,os

#print(tf.__version__)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #忽略弹出的警告。
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


#加载数据
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()  #(60000,28,28),(60000,)  这个数据格式基本与mnist数据格式一样

#print(train_images[0])  #图片的取值范围也是0-255,而不是二值化。
#对数据归一化
train_images = train_images / 255.0   #归一化，注意是处以255.0，这样就是小数了
test_images = test_images / 255.0


#建立模型，3层结构
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),    #将28,28的数据展开成一维
    keras.layers.Dense(128, activation='relu'),    #隐藏层的结构，128个神经元，激活函数是relu,全连接层。
    keras.layers.Dense(10)                         #第三层结构，输出10个数
])


#对模新进行编译
model.compile(optimizer='adam',                    #优化算法是这个
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


model.fit(train_images, train_labels,verbose=1, epochs=10,validation_split=0.1) #batch_size默认为32,shuffle默认为True,显示的准确性可能是每次更新batch时剩下的数据的准确性。
                                                          #或者validation_split=0.1 指定10%的数据作为验证数据。验证数据是混洗之前x和y数据的最后一部分样本中。
                                                          #verbose=1,输出每个batch的结果，但是不知道为啥输出不出来，可能是太快了，可以重定向到文件：>> test.txt
                                                          #虽然不能看到每个batch,但是大致上是能看出来点东西的


test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2) #用测试数据集测试训练模型的结果。
                                                                           #返回值不固定，如果在complie的时候没有指定metrics的话，默认只有loss一个返回值。

#print('\nTest accuracy:', test_acc)

#print (help(model.evaluate))












