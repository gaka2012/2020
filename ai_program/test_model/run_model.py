# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 16:13:26 2020

@author: 12248
"""
import pickle
import gzip,cv2
import numpy as np
from PIL import Image

def load_data():
    
    f = gzip.open('mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
    f.close()
    return (test_data)

def load_data_wrapper():

    te_d = load_data()
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    return (test_data)

def sigmoid(z): #将w和b的结果变成西格玛函数
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def cal_input(weight,biase,a): #a是输入的数据，返回经过神经网络的结果。
    """Return the output of the network if ``a`` is input."""
    
    for b, w in zip(biase, weight):
        a = sigmoid(np.dot(w, a)+b)
    return a
def evaluate_input(weight,biase,test_data): 
    
    test_results = [(np.argmax(cal_input(weight,biase,x)), y) for (x, y) in test_data]  #每一对x,y都会返回一组(x1,y),其中x1是经过神经网络后的返回值。然后下面判断x1,y
                                                                                        #是否相等。
    
    return sum(int(x == y) for (x, y) in test_results) #如果实际结果于理论结果相同，则等于1，这是个元组，里面是很过个1，求和。

#以下程序会加载保存的weight和biases,然后测试test_data，看一下准确率是否与原始程序一致，只需要修改weight和biases的位置。
'''
#加载保存下来的第29次迭代的w和b
r_weight = np.load('/home/zhangzhipeng/software/github/2020/ai_program/save_weight/weight.npz',allow_pickle=True)
r_biases = np.load('/home/zhangzhipeng/software/github/2020/ai_program/save_weight/biases.npz',allow_pickle=True)
out_weight = list(r_weight['out_w']) #将读取的weight转成list格式。
out_biases = list(r_biases['out_b'])


#加载测试数据,并将测试数据list化
test_data  = load_data_wrapper()
test_data  = list(test_data)


test_out   = evaluate_input(out_weight,out_biases,test_data)
print("when epoch ==2  w and b : {} / 10000".format(test_out))
'''



#以下程序会加载保存的weight和biases，然后输入一张图片，看输出结果。

#加载保存下来的第29次迭代的w和b
r_weight = np.load('/home/zhangzhipeng/software/github/2020/ai_program/save_weight/weight.npz',allow_pickle=True)
r_biases = np.load('/home/zhangzhipeng/software/github/2020/ai_program/save_weight/biases.npz',allow_pickle=True)
out_weight = list(r_weight['out_w']) #将读取的weight转成list格式。
out_biases = list(r_biases['out_b'])


'''
#处理图像，首先读入一个彩色的刚切割的图，然后灰度化，更改大小，二值化。
im       = Image.open('333.png')
ori_data = list(im.getdata())
pro_data = [(255-x)*1.0/255.0 for x in ori_data] #因为二值化后的图片
data     = np.reshape(pro_data,(28,28))
print (data)
result   = np.argmax(cal_input(out_weight,out_biases,data))
print(result)
'''

#im       = Image.open('777.png')
#ori_data = list(im.getdata())
#pro_data = [(x-255)*1.0/255.0 for x in ori_data] #因为二值化后的图片

data     = cv2.imread('000.png',0)
np.set_printoptions(linewidth=120)
print (data)


data     = np.reshape(data,(784,1))
data     = data/255
#print (data)
result   = np.argmax(cal_input(out_weight,out_biases,data))
print(result)











