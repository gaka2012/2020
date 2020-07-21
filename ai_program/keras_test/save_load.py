#!/usr/bin/python
# -*- coding:UTF-8 -*-


from __future__ import absolute_import, division, print_function, unicode_literals

import os

import tensorflow as tf
from tensorflow import keras

print(tf.version.VERSION)



#得到数据，mnist中的前1000个数据(节省时间),将数据重新组合成784个，并归一化。
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data() #(60000,28,28),(60000,)  这个数据格式基本与mnist数据格式一样

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0


# 定义一个简单的序列模型
def create_model():
  model = tf.keras.models.Sequential([
    keras.layers.Dense(512, activation='relu', input_shape=(784,)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation='softmax')
  ])

  model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

  return model

# 创建一个基本的模型实例
model = create_model()

# 显示模型的结构
model.summary()

