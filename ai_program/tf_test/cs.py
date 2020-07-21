#!/usr/bin/python
# -*- coding:UTF-8 -*-


from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow.keras import Model, layers
import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #忽略弹出的警告。


#hello = tf.constant("hello world")  #定义一个常量。包含三个属性: 值、shape、dtype
#print(hello)
#print(hello.numpy())

'''
a = tf.constant(2)                   #定义一个整型常数，并进行基本的运算
b = tf.constant(3)
c = a+b                              #加减乘除运算

aa = tf.cast(a,tf.float32)           #数据类型转换，默认是int32,现在转化成了float32
#print (aa)

mean = tf.reduce_mean([a, b,c])      #求均值，会返回整数，没有小数。如果是矩阵则可以指定维度。
su   = tf.reduce_sum([a,b,c])        #求和


x = [[1,2,3],
      [1,2,3]]
      
xx = tf.cast(x,tf.float32)

su = tf.reduce_sum(xx,axis=0)         #计算矩阵的和，axis=0表示计算纵轴。
print (su.numpy())
'''

'''
matrix1 = tf.constant([[1., 2.], [3., 4.]])  #矩阵相乘。
matrix2 = tf.constant([[5., 6.], [7., 8.]])

product = tf.matmul(matrix1, matrix2)
print (product)
'''

a = [1,2,3,4,5,6,7,8]
b = [0,1,2,3,4,5,6,7]

aa = np.reshape(a,(8,1))
bb = np.reshape(b,(8,1))

train_data = tf.data.Dataset.from_tensor_slices((aa,bb))  #这个函数沿输出的数据的第一个维度进行切片 
#for i,j in train_data: #检查数据，train_data中有2个数据，一个是aa，一个是bb
#    print (i.numpy(),j.numpy())

train_data = train_data.repeat(1).shuffle(5).batch(2).prefetch(1) #repeat(1)原始数据只重复一次，每次打乱顺序取5个数值，然后从这5个中随机取2个，最终有4组
                                                                  #repeat(2)相当于有2份原始数据，然后每次打乱顺序取5个数值，然后从这5个中随机取2个，最终有8组。
                                                                  #repeat() 相当于有无数份数据，然后每次打乱顺序取5个数值，然后从这5个中随机取2个
for i,j in train_data: #检查数据，train_data中有2个数据，一个是aa，一个是bb
    print (i.numpy())
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


