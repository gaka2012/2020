#!/usr/bin/python
# -*- coding:UTF-8 -*-


import numpy as np
import cv2


def vectorized_result(j):  ##将答案中的数字改为10维的，
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


#1.2 读取
f = np.load('mnist.npz')
#显示npz文件中有几个目录
catalog_names = f.files #4个目录，['x_test', 'x_train', 'y_train', 'y_test']

#print (catalog_names)



x_train,y_train = f['x_train'], f['y_train']  #60000,28,28; 60000,#图片中的数据分为0和大于0的值，0部分是黑色，其他数值表现大约为白色，不知道为啥不直接弄成0和1
x_test,y_test   = f['x_test'], f['y_test']    #10000个测试数据。


#将数据中大于0的设为255,剩下的设为0
for i in range(60000):
    ret, thresh_img = cv2.threshold(x_train[i],1,255,cv2.THRESH_BINARY) # 二值化,返回的thresh_img是28×28的矩阵。
    x_train[i]      = thresh_img
    
for i in range(10000):
    ret, thresh_img = cv2.threshold(x_test[i],1,255,cv2.THRESH_BINARY) # 二值化,返回的thresh_img是28×28的矩阵。
    x_test[i]      = thresh_img


for i in range(10):
    print (y_test[i])
    cv2.imwrite('{}.png'.format(y_test[i]),x_test[i])

#c = x_train[0]/255
#cv2.imwrite('11.png',c)
'''
training_inputs = [np.reshape(x, (784, 1)) for x in x_train] #输入的训练数据改成784,1格式的。
training_results = [vectorized_result(y) for y in y_train]   #输入的训练答案改成10维的。
training_data = zip(training_inputs, training_results)

test_inputs = [np.reshape(x, (784, 1)) for x in x_test]
test_data = zip(test_inputs, y_test)
'''
 

    
    
    
    
    
