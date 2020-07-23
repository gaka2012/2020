#!/usr/bin/python
# -*- coding:UTF-8 -*-


from __future__ import absolute_import, division, print_function, unicode_literals

import os

import tensorflow as tf
from tensorflow import keras

print(tf.version.VERSION)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #忽略弹出的警告。
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

#得到数据，mnist中的前1000个数据(节省时间),将数据重新组合成784个，并归一化。
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data() #(60000,28,28),(60000,)  这个数据格式基本与mnist数据格式一样

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0


# 定义一个简单的序列模型
def create_model():
  model = tf.keras.models.Sequential([
    keras.layers.Dense(512, activation='relu', input_shape=(784,)),   #512个神经元，激活函数。
    keras.layers.Dropout(0.2),                                        #正则化，丢弃概率是0.2
    keras.layers.Dense(10, activation='softmax')                      #不同层可以使用不同的激活函数吗？
  ])

  model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

  return model

'''
# 创建一个基本的模型实例
model = create_model()

# 显示模型的结构
model.summary()


#创建回调路径
checkpoint_path = "/home/zhangzhipeng/software/github/2020/ai_program/keras_test/w&b/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)



# 创建一个保存模型权重的回调
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

#使用新的回调训练模型
model.fit(train_images, 
          train_labels,  
          epochs=10,
          validation_data=(test_images,test_labels), #指定验证数据
          callbacks=[cp_callback])  # 通过回调训练



#到此为止，上面的程序每个epoch会显示三行，第一行是实际的训练，第二行是saving model，第三行是测试数据集的准确度。
'''


# 创建一个基本模型实例
model = create_model()

# 评估模型
loss, acc = model.evaluate(test_images,  test_labels, verbose=2)
print("Untrained model, accuracy: {:5.2f}%".format(100*acc))



checkpoint_path = "/home/zhangzhipeng/software/github/2020/ai_program/keras_test/w&b
# 加载权重
model.load_weights(checkpoint_path)

# 重新评估模型
loss,acc = model.evaluate(test_images,  test_labels, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))





























