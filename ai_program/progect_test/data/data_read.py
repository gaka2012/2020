# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 09:11:34 2020

@author: 12248
"""

import pickle
import gzip
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import random,glob


#本程序用来将训练数据中的某一组提取出来并画图。
'''
def vectorized_result(j): #将答案中的数字改为10维的，
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

#加载数据
f = gzip.open('mnist.pkl.gz', 'rb')
training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
f.close()

#将输入的数据多维化并形成列表，然后zip.
tr_d, va_d, te_d = training_data,validation_data,test_data

training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]] #50000个数据列表化
training_results = [vectorized_result(y) for y in tr_d[1]]   #将50000个答案标准化到10维，并形成列表
#training_data = zip(training_inputs, training_results)       #将输入的数据和答案zip

#训练数据有784个元素，没有shape,实际上可以将其reshape还原成原始的shape,因为其归于化了，所以乘以255还原。
a = training_data[0][0]
b = a.reshape(28,28)
print ('b',b)
for i in range(28):
    for j in range(28):
        b[i][j]*=255
print (b)

plt.figure(figsize=(25,15))
ax=plt.subplot(1,1,1)
ax.imshow(b)           #将数字化的图片画出来，竟然能原样恢复。
plt.savefig('ori.png')
plt.show()
plt.close()

print (training_results[0])
'''


'''
#本程序用来将自己拍的手写数字转换成28*28像素的图片
from PIL import Image
Image.MAX_IMAGE_PIXELS = 300000000 #读取比较大的图片时，会提示超过其上限，所以重新设置一下上限，这个上限最大不能 超过内存。
import matplotlib.pyplot as plt
import numpy as np


image_path = 'E:\\3.jpg'  #读取图片

img=Image.open(image_path)

#img = img.transpose(Image.ROTATE_270)
img=np.array(img)         #图片数字化
print (img.shape)        
#print (img[:380][:1523][:3])


#将最后的alpha通道的255改为0，就变成白板了。
#newimg = img[0][0][3]
# =============================================================================
# for i in range(380):
#     for j in range(1523):
#         img[i][j][3]=255
# =============================================================================


#提取三维矩阵的一部分
center_h = img.shape[0]//2-1 #只提取照片中的核心的100个像素
center_w = img.shape[1]//2-1

high = [i for i in range(1700,2100)] #根据图片的高度生成范围
widt = [i for i in range(1100,1900)] #根据图片的宽度生成范围
rgb = [0,1,2]
newimg = img[np.ix_(high,widt,rgb)]
#print (newimg)


plt.figure
#根据numpy格式的矩阵画图，可以画原始数据，也可以画去掉alpha通道后的数据的图。
plt.figure(figsize=(2.8,2.8),dpi=10) #figsize中的参数代表的是宽度和高度(英寸)，乘以dpi就是最终生成的实际像素。
ax=plt.subplot(1,1,1)
ax.imshow(newimg)           #将数字化的图片画出来，竟然能原样恢复。
plt.savefig('test1.png')
plt.show()
plt.close()
'''




from PIL import Image
Image.MAX_IMAGE_PIXELS = 300000000 #读取比较大的图片时，会提示超过其上限，所以重新设置一下上限，这个上限最大不能 超过内存。
import matplotlib.pyplot as plt
import numpy as np
import imageio,os


def load_data():
    
    f = gzip.open('D:\\中国地震局第二检测中心\\机器学习\\progect_35\\DeepLearningPython35-master\\mnist.pkl.gz', 'rb')
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
        #print (w.shape)
    return a

def evaluate_input(weight,biase,test_data):
    
    test_results = [(np.argmax(cal_input(weight,biase,x)), y) for (x, y) in test_data[:10]]
    
    return sum(int(x == y) for (x, y) in test_results) #如果实际结果于理论结果相同，则等于1，这是个元组，里面是很过个1，求和。

def recog_figure(weight,biase,data):
    fig_num = np.argmax(cal_input(weight,biase,data))
    return fig_num



image_path = 'E:\\figure\\*.jpg'  #读取图片


images = glob.glob(image_path) #获得所有的图片路径
#img = Image.open
#mg=Image.open(image_path)
#img_L = img.convert("L")


for image in images:
    
    name = os.path.basename(image) #3.jpg
    print(name)
    
    #图片处理
    img = imageio.imread(image,as_gray=True) #直接进行了灰度化，RGB归一到灰色。
    img_data  = 255.0 - img.reshape(784,1)
    img_data  = img_data/255.0
    #img_data = (img_data / 255.0 * 0.99) + 0.01
    #print (img_data.shape)

    #画图
    plt.figure(figsize=(2.8,2.8),dpi=100)
    ax=plt.subplot(1,1,1)
    ax.imshow(img_data.reshape(28,28))           #将数字化的图片画出来，竟然能原样恢复。
    #plt.savefig('new.png')
    plt.show()
    plt.close()

    #加载保存下来的第29次迭代的w和b
    r_weight = np.load('D:\\中国地震局第二检测中心\\机器学习\\progect_35\\out\\weight.npz') #读取保存的weights
    r_biases = np.load('D:\\中国地震局第二检测中心\\机器学习\\progect_35\\out\\biases.npz') #读取保存的biases
    out_weight = list(r_weight['out_w']) #将读取的weight转成list格式。
    out_biases = list(r_biases['out_b'])

    result = recog_figure(out_weight,out_biases,img_data)
    print ('result==',result)







#img_L=np.array(img_L)         #图片数字化
#img = np.array(img)
#print (img.shape)

'''
high = [i for i in range(28)] #根据图片的高度生成范围
widt = [i for i in range(28)] #根据图片的宽度生成范围
rgb = [0,1,2]
newimg = img[np.ix_(high,widt,rgb)]  #图片的数据格式是unit8的，估计跟int32差不都，不用管，直接转换。
newimg = newimg.astype(np.float64)
print (newimg.dtype)

#灰度化，归一化,28*28维度化
new_list = []
for i in range(28):
    for j in range(28):
        gray_num = img[i][j][0]*0.299+img[i][j][1]*0.587+img[i][j][2]*0.114
        
        new_list.append(gray_num)
out = np.reshape(new_list,(28,28))
print (out)
'''





'''
for i in range(28):
    for j in range(28):
        img_L[i][j]/=255.0
print (img_L)

# =============================================================================
# high = [i for i in range(28)] #根据图片的高度生成范围
# widt = [i for i in range(28)] #根据图片的宽度生成范围
# rgb = [0,1,2]
# newimg = img_L[np.ix_(high,widt,rgb)]
# print (newimg)
# =============================================================================



plt.figure(figsize=(25,15),dpi=10)
ax=plt.subplot(1,1,1)
ax.imshow(img_L)           #将数字化的图片画出来，竟然能原样恢复。
plt.savefig('test2.png')
plt.show()
plt.close()


'''



'''

def load_data():
    
    f = gzip.open('D:\\中国地震局第二检测中心\\机器学习\\progect_35\\DeepLearningPython35-master\\mnist.pkl.gz', 'rb')
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
        #print (w.shape)
    return a

def evaluate_input(weight,biase,test_data):
    
    test_results = [(np.argmax(cal_input(weight,biase,x)), y) for (x, y) in test_data[:10]]
    
    return sum(int(x == y) for (x, y) in test_results) #如果实际结果于理论结果相同，则等于1，这是个元组，里面是很过个1，求和。

def recog_figure(weight,biase,data):
    fig_num = np.argmax(cal_input(weight,biase,data))
    return fig_num


#加载保存下来的第29次迭代的w和b
r_weight = np.load('D:\\中国地震局第二检测中心\\机器学习\\progect_35\\out\\weight.npz') #读取保存的weights
r_biases = np.load('D:\\中国地震局第二检测中心\\机器学习\\progect_35\\out\\biases.npz') #读取保存的biases
out_weight = list(r_weight['out_w']) #将读取的weight转成list格式。
out_biases = list(r_biases['out_b'])

#print (out_weight[0].shape)

#加载测试数据
test_data  = load_data_wrapper()
test_data  = list(test_data)

test_out   = evaluate_input(out_weight,out_biases,test_data)
print(test_out)
'''












'''
print(len(training_inputs))

e = np.zeros((10, 1))
#e[j] = 1.0
print(e.shape)
'''
'''
def sigmoid(z): #将w和b的结果变成西格玛函数
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z): #西格玛函数求导
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))
x=[1,2,3,4,5]
y=[9,8,7,6,0]

x = np.reshape(x,(5,1))

print(x)
print(x.transpose())
'''

'''
x_y=zip(x,y) #生成的zip函数并不是列表模式，所以不能用切片的方法，直接输出也不行。只能用下面的迭代的方法。

x_y = list(x_y)

random.shuffle(x_y)
print (len(x_y))
for i in x_y:
    print (i,type(i))


training_data = zip(training_inputs, training_results)
validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
validation_data = zip(validation_inputs, va_d[1])
test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
test_data = zip(test_inputs, te_d[1])
#return (training_data, validation_data, test_data)
'''



















'''
#本程序用来挑选每月的30%的图片，需要更改fig_path，然后会按照每三个图片过滤，只保留第一个，剩下的2个直接删除。

import glob,os,shutil,sys

fig_path = 'E:\\YUN\\KM\\*'

fig_year = sorted(glob.glob(fig_path))
total    = len(fig_year)
num      = 1

for year in fig_year:
    
    #写进度条
    percent=num/total #用于写进度条
    sys.stdout.write("\r{0}{1}".format("#"*10 , '%.2f%%' % (percent * 100)))
    sys.stdout.flush()
    num+=1
    
    mon = sorted(glob.glob(year+'\\*'))
    for png in mon: #E:\YUN\KM\1996\07
        name   = sorted(glob.glob(png+'\\*.png')) #统计某个月份的所有图片
        
        #遍历某个月份下的图片，只保留30%，剩下的删除。
        i = 0 #计数所有图片
        l = 0 #技术剩下的图片
        for fig in name:
            if i%3==0: #如果对3取余等于0则跳过，否则直接删除。
                name = os.path.basename(fig)
                l +=1
            else:
                pass
                #os.remove(fig)
            i+=1  
        
        print (png.split('\\')[1:])
        #print ('{} 共{}张图纸，抽查图纸方式A，共抽查图纸{}张。'.format())
           
        fa = open('result.txt','a+')
        fa.write('{} 共{}张图纸，抽查图纸方式A，共抽查图纸{}张。'.format())
        fa.close()
       
'''





'''
os.chdir(fig_path)
fig_name = glob.glob(fig_path+'\\*.png')
fig_name = sorted(fig_name)

total = len(fig_name)
num = 0
i=1
for fig in fig_name:
    
    percent=i/total #用于写进度条
    sys.stdout.write("\r{0}{1}".format("#"*10 , '%.2f%%' % (percent * 100)))
    sys.stdout.flush()
    
    if i%3==0: #如果对3取余等于0则跳过，否则直接删除。
        name = os.path.basename(fig)
        num +=1
        #print (name)
    else:
        os.remove(fig)
    i+=1  

print ()
print ('left {} pictures'.format(num))
'''
















