#!/usr/bin/python
# -*- coding:UTF-8 -*-

import os,time,threading,random,shutil,cv2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np
import pandas as pd
import threading,time 
import matplotlib.pyplot as plt
import matplotlib.image as mping
from data_reader import Config, DataReader, DataReader_test, DataReader_pred, DataReader_mseed
from tqdm import tqdm
from calss_test import Person



#标签1-1.1 P170查看batch数据集中的一张图,数据集不是bin版本
'''
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

data_read = unpickle('./book_data/cifar-10-batches-py/data_batch_1') #得到的是一个字典

#获取所有的键：[b'batch_label', b'labels', b'data', b'filenames']
keys = list(data_read.keys())

#label标签有10000个,范围是0-9,对应10个分类。
value_1 = list(data_read[b'labels']) 

#data标签中是一堆数，
value_2 = list(data_read[b'data'])
png_0   = value_2[3]  #展示一下第一个图,shape是3072, 32x32的3通道的彩色图。

image_m = np.reshape(png_0,(3,32,32))  #不能是32,32,3 会打乱原来的排列顺序
                                       #不能直接画图，因为只识别32,32,3这个不是别，最后一个必须是通道数

r = image_m[0,:,:]                     #提取红绿蓝，然后融合，画图
g = image_m[1,:,:]
b = image_m[2,:,:]

img23 = cv2.merge([r,g,b])
 
plt.figure()
plt.imshow(img23)
plt.show()

#for key,value in data_read.items():
#    print (key)
'''    


'''
#标签1-1.2 P173 批量导入CIFAR数据集并处理，数据集是bin后缀
import cifar10_input
import pylab

#调用函数取数据
batch_size = 128   #一次读取128个数据，
data_dir   = './book_data/cifar-10-batches-bin'
images_test,labels_test = cifar10_input.inputs(eval_data=False,data_dir=data_dir,batch_size=batch_size)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
tf.train.start_queue_runners()
image_batch, label_batch = sess.run([images_test, labels_test])  #得到的数据shape是(128, 24, 24, 3),标签shape是(128,)

print (label_batch.shape)
#print("__\n",image_batch[0])

#print("__\n",label_batch[0])
#pylab.imshow(image_batch[0])
#pylab.show()

'''











'''
#P166卷积与反卷积

#1.1读取图片
my_img = mping.imread('bike.jpg')
#plt.imshow(my_img)
#plt.axis('off')
#plt.show()
print (my_img.shape)



#1.2 卷积
full =  np.reshape(my_img,[1,546,820,3])
input_x = tf.Variable(tf.constant(1.0,shape=[1,546,820,3])) #后面会喂数据
#input_x = tf.placeholder(dtype=uint8,shape=[1,546,820,3])

#filter1 = tf.Variable(tf.constant([[-1.0,-1.0,-1.0],[0,0,0],[1.0,1.0,1.0],[-1.0,-1.0,-1.0],[0,0,0],[1.0,1.0,1.0],[-1.0,-1.0,-1.0],[0,0,0],[1.0,1.0,1.0]],shape=[3,3,3,1]))
#filter2 = tf.Variable(tf.constant([[-1.0,-1.0,-1.0],[0,0,0],[1.0,1.0,1.0]],shape=[3,3]))
filter3 = tf.contrib.layers.xavier_initializer()  #生成一个随机的初始化的滤波器，主要用于下面的tf.layers.conv2d传递初始化参数。

#op      = tf.nn.conv2d(input_x,filter1,strides=[1,1,1,1],padding='SAME')
op      = tf.layers.conv2d(input_x,filters=3,kernel_size=[3,3],activation=None,padding='SAME',kernel_initializer=filter3) #3个滤波器就可以获得3个不同的滤波后的特征图
o       = tf.cast( ((op-tf.reduce_min(op))/(tf.reduce_max(op)-tf.reduce_min(op)))*255,tf.uint8)                           #归一化到0-255,就可以成图。 

#op1      = tf.layers.conv2d(o,filters=1,kernel_size=[3,3],activation=None,padding='SAME',kernel_initializer=filter3)
#o1       = tf.cast( ((op1-tf.reduce_min(op1))/(tf.reduce_max(op1)-tf.reduce_min(op1)))*255,tf.uint8)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    t = sess.run([o],feed_dict={input_x:full})
    
    t = np.reshape(t,[546,820,3])
    t1 = t[:,:,0]
    plt.imshow(t1,cmap='Greys_r')
    plt.axis('off')
    #plt.show()
    name = str(random.randint(0,10000))
    plt.savefig(name+'_01')
    
    t2 = t[:,:,1]
    plt.imshow(t2,cmap='Greys_r')
    plt.axis('off')
    #plt.show()
    name = str(random.randint(0,10000))
    plt.savefig(name+'_02')
    
    t3 = t[:,:,2]
    plt.imshow(t3,cmap='Greys_r')
    plt.axis('off')
    #plt.show()
    name = str(random.randint(0,10000))
    plt.savefig(name+'_03')
    
'''
    
'''
#initializer = tf.contrib.layers.xavier_initializer()  #返回值：该函数返回一个用于初始化权重的初始化程序 “Xavier，初始化权重矩阵，
#net = tf.layers.conv2d(input_x,
                   filters=1,   #1个滤波器
                   kernel_size=[3,3],  #滤波器大小,默认步长是[1,1]
                   activation=None,
                   padding='same',
                   kernel_initializer=initializer,  #通过上面的初始化方法，返回一个初始化权重矩阵。
                   #kernel_regularizer=self.regularizer, #正则化默认为None
                   #bias_regularizer=self.regularizer,
                   name="input_conv")

#result = tf.cast( ((net-tf.reduce_min(net))/(tf.reduce_max(net)-tf.reduce_min(net)))*255,tf.uint8)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #t,f = sess.run([result,initializer],feed_dict={input_x:full})
    #t   = np.reshape(t,[])
    t = 
    print (t.shape)

'''


#tqdm的使用
'''
progressbar = tqdm(range(0, 100, 2))

for step in progressbar:
    print (step)  #0,2,4,6,8
'''


#因为有时候我们要提取的特征非常多非常广泛，所以需要我们用更多的矩阵来扫（多扫几遍




'''
input1  = tf.Variable(tf.constant(1.0,shape=[1,5,5,1])) #一个批次的数量，图片高度，图片宽度，通道数
#filter1 = tf.Variable(tf.constant([-1.0,0,0,-1],shape=[2,2,1,1])) 
#filter1 = tf.Variable(tf.constant([[1.0,0.0],[2.0,2.0],[1.0,0.0],[2.0,2.0],[1.0,0.0],[2.0,2.0]],shape=[2,2,3,2]))  #这样写最后的结果就是单个卷积核的形式是[1,0],[2,2]，共
#                                                                                                                   #三个相同的卷积核， 
filter2 = tf.Variable(tf.constant([-1.0,0,0,-1,-1,0,0,-1],shape=[2,2,1,2]))
#op1     = tf.nn.conv2d(input1,filter2,strides=[1,2,2,1],padding='VALID')                     #
op1      = tf.layers.conv2d(input1,filters=2,kernel_size=[2,2],strides=(2, 2),activation=None,padding='VALID')  #这样输出的是2个feature map,是一样的。
init    = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    #print (sess.run([op1,filter1]))
    #print (sess.run(filter1))
    print ('---------')
    print (sess.run(op1))
'''








#a = [1,2,3,4,5,6,7,8]
#b = np.reshape(a,[4,2,1,1]) #重新定义成2行3列。

#print (b)

#感受野的计算 https://blog.csdn.net/program_developer/article/details/80958716  #在公式中，最后一层表示为i+1,其前一层表示为i


#https://blog.csdn.net/gqixf/article/details/80519912
#tf.nn.conv2d与tf.layers.conv2d的区别
#res = tf.layers.conv2d(x,filters=32,kernel_size=[3,3],strides=1,padding='SAME')
#即用32个3*3的卷积核对x进行卷积操作，与tf.nn.conv2d不同的是，tf.nn.conv2d不仅需要对权重初始化，还需要定义卷积核的维度。例如，如果我要用64个3*3的卷积核对大小为256*256*32的feature map进行卷积的话，卷积核需设置为[3,3,32,64]，

#参数：dilation_rate,默认为(1,1)，即不扩张   https://www.cnblogs.com/zf-blog/p/12101300.html    

#tf.layers.batch_normalization 
#https://www.cnblogs.com/fclbky/p/12636842.html
#https://blog.csdn.net/TheHonestBob/article/details/103951083

#tf.layers.conv2d_transpose
'''
#书中P131的例子

learning_rate = 1e-4
n_input = 2 #输入层2个节点
n_label = 1
n_hidden = 2 #隐藏层节点个数

x = tf.placeholder(tf.float32,[None,n_input])
y = tf.placeholder(tf.float32,[None,n_label])

weights = {
    'h1':tf.Variable(tf.truncated_normal([n_input,n_hidden],stddev=0.1)),
    'h2':tf.Variable(tf.truncated_normal([n_hidden,n_label],stddev=0.1))
          }

biases = {
     'h1':tf.Variable(tf.zeros([n_hidden])),
     'h2':tf.Variable(tf.zeros([n_label]))
         }

layer_1 = tf.nn.relu(tf.add(tf.matmul(x,weights['h1']),biases['h1']))
y_pred  = tf.nn.tanh(tf.add(tf.matmul(layer_1,weights['h2']),biases['h2']))

loss = tf.reduce_mean((y_pred-y)**2)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)


#生成数据
X = [[0,0],[0,1],[1,0],[1,1]]
Y = [[0],[1],[1],[0]]
X = np.array(X).astype('float32')
Y = np.array(Y).astype('int16')

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

for i in range(10):
    sess.run(train_step,feed_dict={x:X,y:Y})
    print (sess.run(biases['h2']))   

#print (sess.run(y_pred,feed_dict={x:X}))

#print (sess.run(layer_1,feed_dict={x:X}))
'''





'''
#书中P120页的例子
def generate(sample_size,mean,cov,diff,regression):
    num_classes=2
    samples_per_class = int(sample_size/2)
    
    X0 = np.random.multivariate_normal(mean,cov,samples_per_class) #500,2 符合正态分布，不用管写的那么复杂。
    Y0 = np.zeros(samples_per_class)  #500个0
    
    for ci,d in enumerate(diff):
        
        X1 = np.random.multivariate_normal(mean+d,cov,samples_per_class) #
        Y1 = (ci+1)*np.ones(samples_per_class)   #500个1
        
        X0 = np.concatenate((X0,X1))  #数组的拼接
        Y0 = np.concatenate((Y0,Y1))
    
    if regression == False:
        class_ind = [Y0==class_number for class_number in range(num_classes)]  
        Y         = np.asarray(np.hstack(class_ind),dtype=np.float32)           #本来是1000个数，前500个是0,后500个是1.现在变成前500个True,后500个0 ; 然后500个是False,再T

    #X,Y = np.shuffle(X0,Y0)
    X,Y = (X0,Y0)  #Y.shape = (1000,)
    
    return X,Y
    
    
np.random.seed(10)  #设置随机数种子，这样每次生成的随机数都是一样的，里面的参数是随便设置的，没有影响。
num_classes = 2
mean = np.random.randn(num_classes)  #具有正态分布

cov = np.eye(num_classes)  #对角矩阵

#X,Y = generate(1000,mean,cov,[3.0],True)

#colors = ['r' if l==0 else 'b' for l in Y[:]]


#plt.scatter(X[:,0],X[:,1],c=colors)
#plt.xlabel("Scaled age (in yrs)")
#plt.ylabel("Tumor size (in cm)")
#plt.show()
lab_dim = 1
input_dim = 2







#标准输入及其答案
input_features = tf.placeholder(tf.float32,[None,input_dim]) 
input_labels   = tf.placeholder(tf.float32,[None,lab_dim])

#定义学习参数
W = tf.Variable(tf.random_normal([input_dim,lab_dim]),name='weight')
b = tf.Variable(tf.zeros([lab_dim]),name='bias')

output = tf.nn.sigmoid(tf.matmul(input_features,W)+b)

#交叉熵 https://blog.csdn.net/hellocsz/article/details/91347592
cross_entropy = -(input_labels * tf.log(output)+(1-input_labels)*tf.log(1-output))

loss          = tf.reduce_mean(cross_entropy)   #求一个batch的交叉熵的均值

#https://blog.csdn.net/dcrmg/article/details/79797826  tf.reduce_mean：求均值 
ser  = tf.square(input_labels-output)
err  = tf.reduce_mean(ser)

optimizer = tf.train.AdamOptimizer(0.04)
train     = optimizer.minimize(loss)

maxEpochs = 50
minibatchSize = 25



with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    #迭代次数，迭代50次数
    for epoch in range(maxEpochs):
        sumerr = 0
        
        #每次迭代训练都只提取一个batch进行训练
        for i in range(np.int32(len(Y)/minibatchSize)):  #每训练一次，比如100个数据，要分成4组。
            x1 = X[i*minibatchSize:(i+1)*minibatchSize,:]
            y1 = np.reshape(Y[i*minibatchSize:(i+1)*minibatchSize],[-1,1])
            
            tf.reshape(y1,[-1,1])
            _,lossval,outputval,errval = sess.run([train,loss,output,err],feed_dict={input_features:x1,input_labels:y1})
            sumerr = sumerr+errval
        
        print ("Epoch:",'%04d'%(epoch+1),"cost=","{:.9f}".format (lossval),"err=",sumerr/np.int32(len(Y)/minibatchSize))
'''








'''
#书中P161的例子 
input1  = tf.Variable(tf.constant(1.0,shape=[1,5,5,1])) #一个批次的数量，图片高度，图片宽度，通道数
#filter1 = tf.Variable(tf.constant([-1.0,0,0,-1],shape=[2,2,1,1])) 
filter1 = tf.Variable(tf.constant([[1.0,0.0],[2.0,2.0]],shape=[2,2,2,1])) 
#op1     = tf.nn.conv2d(input1,filter1,strides=[1,2,2,1],padding='SAME')

init    = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    #print (sess.run([op1,filter1]))
    print (sess.run(filter1))


#书中P166的例子
'''








'''
#1.1写入
#将数据村成npz格式的数据,默认的名字是arr_0,arr_1,arr_2等名字，也可以自定义为c_array
a = np.arange(3)
b = np.arange(4)
c = 12.0  #本来是整性，转换成npz后就变成了numpy
np.savez('test.npz',a,b,c_array=c)
print (a,b,c,type(c))

#1.2 读取
#A = np.load('test.npz')
#显示npz文件中有几个目录
catalog_names = A.files

print (catalog_names)
#print (type(A[catalog_names[2]]))


a = np.array([1,2,3,4,5,6,7,8,9])

b = np.reshape(a,[3,3]) #重新定义成2行3列。
c = b[:,np.newaxis,:]  #增加一个维度，shape变成了(5,1)
print (c,c.shape)
np.savez('./sample/11.npz',c)

A = np.load('./sample/11.npz')
name = A.files
data = A[name[0]]
print (data.shape)

a = 'new'

b = './test/{}'.format(a)

print (b)
'''

'''
hcoord = tf.train.Coordinator()
data_reader = DataReader(data_dir  = './test_data',
                         data_list = './test_data.csv',
                         mask_window = 0.4,
                         queue_size = 200*3,
                         coord = coord)
    

data_process = data_reader.start_threads()
#print (data_process)
'''
'''
a = np.array([1,2,3,4,5,6,7,8,9,10,11,12])

b = np.reshape(a,[4,3]) #重新定义成2行3列。
c = b[:,np.newaxis,:]  #增加一个维度，shape变成了(5,1)


data = c[:,0,1].flatten()
print (c.shape)
print (data)
print (data.shape)
'''

'''
#取均值
data = np.mean(c,axis=0,keepdims=True)
data = c-data

std_data = np.std(data,axis=0,keepdims=True)
print (std_data)

data /= std_data
print (data.shape)
print (data)



a = np.array([1,2,3,4,5,6,7,8,9,10,11,12])

b = np.reshape(a,[4,3]) #重新定义成2行3列。
c = b[:,np.newaxis,:]  #增加一个维度，shape变成了(5,1)

print (c.shape)

a1 = np.array([1,2,3])
b1 = np.reshape(a1,[1,3])
c1 = b1[:,np.newaxis,:]
print (c1,c1.shape)

data = c/c1
print (data,data.shape)
'''







'''
with tf.variable_scope('name1',):
    var1 = tf.get_variable('firstvar',shape=[2],dtype=tf.float32)
     
with tf.variable_scope('name2',):                                  #作用域
    var2 = tf.get_variable('firstvar',shape=[2],dtype=tf.float32)  #正常情况下，get_variable不能指定相同的名字，但是在作用域的情况下，是可以相同的。
                                                                   #支持嵌套，P66
print (var1.name,var2.name)

'''







'''
#meta = np.load('NC_MEM_2017100709282692.npz')
#data = meta['data']
#itp  = meta['itp']
#its  = meta['its']

#start_tp = meta['itp'].tolist()
#print (itp,start_tp)

a = [1,2,3,4,5,6,7,8]
b = np.reshape(a,[4,2]) #重新定义成2行3列。
c = b[:,np.newaxis,:]
#print (c.shape,c)

'''

#print (c[:,0,0])

#np.exp(-(np.arange(-self.mask_window//2,self.mask_window//2))**2/(2*(self.mask_window//4)**2))[:target.shape[0]-(itp-self.mask_window//2)]
#np.exp(-(np.arange(-40//2,40//2))**2/(2*(40//4)**2))[:3000-(itp-20)]

#test = (-np.arange(-40//2,40//2)**2/200)[:100] #[:3000-(60-20)] #前面生成一个numpy格式的数据，后面相当于去提取前面生成的数据。类似与 a=[1,2,3]  b=a[0:2]
#test = np.exp(np.arange(-2,2))[:100]

#test1 = np.exp(test)[:2880]

#print (test,test.shape)

'''
from multiprocessing import Pool

import multiprocessing
n_threads = multiprocessing.cpu_count()  #看看有几个cpu
print (n_threads)

#多线程教程。
def saySorry(b):
    c = 0
    for i in range(4*20000000):
        c+=b
    
    #print(b,threading.current_thread().name)
    print ('test',b)


for i in range(5):
    # 创建一个线程 
    #mthread = threading.Thread(target=function_name, args=(function_parameter1, function_parameterN)) 
    mthread = threading.Thread(target = saySorry,args=(2,3))
    # 启动刚刚创建的线程 
    mthread .start()
#mthread.join()  #注意这个方法所在的位置，如果在这个位置则同时运行上面的5个线程，然后运行下面的。
                #如果在for循环中，则相当于每次都阻塞，线程就失去作用了。
                #也可以对其添加一个时间，这样就只会阻塞一定的时间。



pool = Pool()
para = [5,6,7,8,0,1,2,3]
out  = pool.map(saySorry,para)


print ('我是分割线')

for i in range(5):
    saySorry(4)

'''










'''
a = tf.Variable(tf.zeros([1,2]))   #定义变量，定义到流程图中。
b = tf.Variable(tf.ones([1,2]))
c = a+b 

init = tf.global_variables_initializer()  #初始化流程图，并运行。
sess = tf.Session()

sess.run(init)

print (sess.run(a))                       #查看a得值。 
print (sess.run(b))



#保存和加载模型
#https://blog.csdn.net/qq_37764129/article/details/93069843?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-1.channel_param&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-1.channel_param
#Tensorflow 会自动生成4个文件,
#第一个文件为 model.ckpt.meta，保存了 Tensorflow 计算图的结构，可以简单理解为神经网络的网络结构。
#model.ckpt.index 和 model.ckpt.data-*****-of-***** 文件保存了所有变量的取值。
#最后一个文件为 checkpoint 文件，保存了一个目录下所有的模型文件列表。


# 1.准备数据： 
x = tf.placeholder(tf.float32, shape=[None, 1])  
y = 4 * x + 4  #需要拟合的函数公式，系数为4,常数为4
 
# 2.构造一个线性模型，系数和常数给一个随机值，然后通过迭代的方式去拟合上面的函数公式 
w = tf.Variable(tf.random_normal([1], -1, 1)) #生成一个随机数
b = tf.Variable(tf.zeros([1]))                #b=0
y_predict = w * x + b


# 3.求解模型
# 设置损失函数：误差的均方差 
loss = tf.reduce_mean(tf.square(y - y_predict))
# 选择梯度下降的方法
optimizer = tf.train.GradientDescentOptimizer(0.5)
# 迭代的目标：最小化损失函数
train = optimizer.minimize(loss)


#参数定义声明 
isTrain = True
train_steps = 100
checkpoint_steps = 50
checkpoint_dir = ''
saver = tf.train.Saver()  # defaults to saving all variables - in this case w and b
x_data = np.reshape(np.random.rand(10).astype(np.float32), (10, 1))  #生成10个[0,1)区间的随机数，然后reshape一下。
 
############################################################
# 以下是用 tf 来解决上面的任务

 
# 设置tensorflow对GPU的使用按需分配
#config  = tf.ConfigProto()
#config.gpu_options.allow_growth = True
 
# 2.启动图 (graph)
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables()) 
    #判断当前工作状态
    if isTrain: #isTrain:True表示训练；False：表示测试
        # 3.迭代，反复执行上面的最小化损失函数这一操作（train op）,拟合平面
        for i in xrange(train_steps): #train_steps表示训练的次数，例子中使用100
            sess.run(train, feed_dict={x: x_data})
            if (i + 1) % checkpoint_steps == 0: #表示训练多少次保存一下checkpoints，例子中使用50
                print ('step: {}  train_acc: {}  loss: {}'.format(step, sess.run(W), sess.run(b)))
                saver.save(sess, checkpoint_dir + 'model.ckpt', global_step=i+1) #表示checkpoints文件的保存路径，例子中使用当前路径
    else: #如果isTrain=False，则进行测试
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path) #恢复变量
        else:
            pass
        print(sess.run(w),sess.run(b))

'''




'''
session_config = tf.ConfigProto(       #配置tf.session的运行方式，比如GPU或者是CPU
      log_device_placement=True,       #打印tensorflow使用了那种操作。
      inter_op_parallelism_threads=0,
      intra_op_parallelism_threads=0,
      allow_soft_placement=True) #它的cpu和gpu是不同的，如果将这个选项设置成True，那么当运行设备不满足要求时，会自动分配GPU或者CPU。

sess = tf.Session(config=session_config)

a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2,3], name='b')
b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3,2], name='b')

c = tf.matmul(a,b)
print(sess.run(c))
'''











 
'''
     
# 样本个数00
sample_num=5
# 设置迭代次数
epoch_num = 2
# 设置一个批次中包含样本个数
batch_size = 3
# 计算每一轮epoch中含有的batch个数
batch_total = int(sample_num/batch_size)+1
 
# 生成4个数据和标签
def generate_data(sample_num=sample_num):
    labels = np.asarray(range(0, sample_num)) #[0,1,2,3,4]
    images = np.random.random([sample_num, 224, 224, 3])
    print('image size {},label size :{}'.format(images.shape, labels.shape))
    return images,labels
 
def get_batch_data(batch_size=batch_size):
    images, label = generate_data()
    # 数据类型转换为tf.float32
    images = tf.cast(images, tf.float32)
    label = tf.cast(label, tf.int32)
 
    #从tensor列表中按顺序或随机抽取一个tensor准备放入文件名称队列,https://blog.csdn.net/dcrmg/article/details/79776876
    input_queue = tf.train.slice_input_producer([images, label], num_epochs=epoch_num, shuffle=False)
 
    #从文件名称队列中读取文件准备放入文件队列
    image_batch, label_batch = tf.train.batch(input_queue, batch_size=batch_size, num_threads=2, capacity=64, allow_smaller_final_batch=False)
    return image_batch, label_batch
 
 
#image_batch, label_batch = get_batch_data(batch_size=batch_size)
labels = np.random.random([2,3,2,2])
print (labels)
 
 

with tf.Session() as sess:
 
    # 先执行初始化工作
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
 
    # 开启一个协调器
    coord = tf.train.Coordinator()
    # 使用start_queue_runners 启动队列填充,承接上面的tf.train.slice_input_producer
    threads = tf.train.start_queue_runners(sess, coord)
 
    try:
        while not coord.should_stop():
            print '************'
            # 获取每一个batch中batch_size个样本和标签
            image_batch_v, label_batch_v = sess.run([image_batch, label_batch])
            print(image_batch_v.shape, label_batch_v)
    except tf.errors.OutOfRangeError:  #如果读取到文件队列末尾会抛出此异常
        print("done! now lets kill all the threads……")
    finally:
        # 协调器coord发出所有线程终止信号
        coord.request_stop()
        print('all threads are asked to stop!')
    coord.join(threads) #把开启的线程加入主线程，等待threads结束
    print('all threads are stopped!')

'''























#Tensorflow依赖于一个高效的C++后端来进行计算。与后端的这个连接叫做session。一般而言，使用TensorFlow程序的流程是先创建一个图，然后在session中启动它。

#这里，我们使用更加方便的InteractiveSession类。通过它，你可以更加灵活地构建你的代码。它能让你在运行图的时候，插入一些计算图，这些计算图是由某些操作(operations)构成的。这对于工作在交互式环境中的人们来说非常便利，比如使用IPython。如果你没有使用InteractiveSession，那么你需要在启动session之前构建整个计算图，然后启动该计算图。
#eval和run都是获取当前结点的值的一种方式。两者的区别主要在于，eval一次只能得到一个结点的值，而run可以得到多个。



#多线程数据输入 https://www.jianshu.com/p/d063804fb272
#包含三步：创建队列(不用run)、将元素输入到队列、取出队列。
#FIFOQueue：先入先出队列--第一个进入的同时也会第一个出来；PaddingFIFOQueue:以固定长度批量出列队列，形式稍微不同，本质一致。
#tf.FIFOQueue(capacity, dtypes, shapes=None, names=None ...)


#QueueRunner：队列管理器；
#Coordinator：协调器。

'''
tf.InteractiveSession()

q = tf.FIFOQueue(2, "float")  #属性是FIFQQueue
x = q.enqueue_many(([9,3],[1,2]))  #属性是Operation

y = q.dequeue()               #属性是Tensor

#init.run()
x.run()
print (y.eval())
y = x+1
q_inc = q.enqueue([y])

init.run()
q_inc.run()
q_inc.run()
q_inc.run()
#x.eval()  # 返回1
#x.eval()  # 返回2
#x.eval()  # 卡住

'''


'''
#格式与data_reader相似的队列操作
a = [1,2,3,4]
b = np.reshape(a,[2,2])

a1 = [2,2,2,2]
b1 = np.reshape(a1,[2,2])

c = [b,b1]
print (type(c))
#建立流程图，包括建立队列，数据进入队列，取出队列中数据
queue   = tf.PaddingFIFOQueue(3,['float32','float32'],shapes=[(2,2),(2,2)])  #创建一个长度为3的队列，数据类型为float,shape为这个
#enqueue = queue.enqueue(b)  #单独一个进入队列
#dequeue = queue.dequeue()   #单独取出一个数列

init    = queue.enqueue([b,b1]) #入列
init1   = queue.dequeue_many(2) #出列

with tf.Session() as sess:
    sess.run(init) #入列初始化
    sess.run(init)
    print (sess.run(init1))  #出列的时候分开了？
'''



'''
#队列操作测试 
with tf.Session() as sess:
    q = tf.FIFOQueue(3, 'float')              #创建一个长度为3的队列，数据类型为float.
    init = q.enqueue_many(([0.1, 0.2, 0.3],)) #将零个或多个元素编入队列之中
    init2 = q.dequeue()                          #把元素从队列中移出，如果在执行该操作时队列已空，那么将会阻塞直到元素出列，返回出列的tensors的tuple
    #init3 = q.dequeue_many((1))                     #tf.FIFOQueue是先入先出的队列，貌似没有批量出列的操作？
    
 
 
 
    sess.run(init)    
    print (sess.run(init3))
    #循环，将所有数据提取出来(出列)，如果range范围超过了队列中的拥有的数据量，则会卡到那个地方
    #for i in range(3):
    #    print (init2.eval())
    
    
    

    #quelen = sess.run(q.size())  #不要被前面的sess.run迷惑了，不用管它，就是一个执行后面命令的指示，实际上要表达的只是q.size而已。
    #for i in range(quelen):
    #    print(sess.run(q.dequeue()))

'''

'''
#https://www.jianshu.com/p/d063804fb272  协调器与队列
#因此通常会使用多个线程读取数据，然后使用一个线程消费数据
q = tf.FIFOQueue(10, "float")  
counter = tf.Variable(0.0)  #计数器

# 给计数器加一
increment_op = tf.assign_add(counter, 1.0)  #更新ref的值，通过增加value，即：ref = ref + value；

# 将计数器加入队列
enqueue_op = q.enqueue(counter)



# 创建QueueRunner
# 用多个线程向队列添加数据
# 这里实际创建了4个线程，两个增加计数，两个执行入队
qr = tf.train.QueueRunner(q, enqueue_ops=[increment_op, enqueue_op])

# 主线程
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
# 启动入队线程
qr.create_threads(sess, start=True)
for i in range(20):
    print (sess.run(q.dequeue()))


#增加计数的进程会不停的后台运行，执行入队的进程会先执行10次（因为队列长度只有10），然后主线程开始消费数据，当一部分数据消费被后，入队的进程又会开始执行。最终主线程消费完20个数据后停止，但其他线程继续运行，程序不会结束。
'''

'''
#承接上面的，简书中的第二个例子
# 子线程函数
def loop(coord, id):
    t = 0
    while not coord.should_stop():  #这个while是与下面的coord.request_stop相对应的
        print(id)                   #在这个while循环中，每次都打印出id来
        time.sleep(1)
        t += 1
        # 只有1号线程调用request_stop方法
        if (t >= 2 and id == 1):    #只要有任何一个线程调用了Coordinator的request_stop方法，所有的线程都可以通过should_stop方法感知并停止当前线程。
            coord.request_stop()

# 主线程
coord = tf.train.Coordinator()

# 使用Python API创建10个线程
threads = [threading.Thread(target=loop, args=(coord, i)) for i in range(4)]

# 启动所有线程，比如开了3个线程，则同时运行3个线程，但是由于for循环也是有先来后到的，因此实际上并不是完全同时的，显示到终端上会发现每次几乎都是0,1,2的顺序(不绝对)。
for t in threads: t.start()

coord.join(threads)

#将这个程序运行起来，会发现所有的子线程执行完两个周期后都会停止，主线程会等待所有子线程都停止后结束，从而使整个程序结束。由此可见，只要有任何一个线程调用了Coordinator的request_stop方法，所有的线程都可以通过should_stop方法感知并停止当前线程。

print ('this is a end')



#简书中的第三个例子
# 1000个4维输入向量，每个数取值为1-10之间的随机数
data = 10 * np.random.randn(1000, 4) + 1
# 1000个随机的目标值，值为0或1
target = np.random.randint(0, 2, size=1000)

# 创建Queue，队列中每一项包含一个输入数据和相应的目标值
queue = tf.FIFOQueue(capacity=50, dtypes=[tf.float32, tf.int32], shapes=[[4], []])

# 批量入列数据（这是一个Operation）
enqueue_op = queue.enqueue_many([data, target])
# 出列数据（这是一个Tensor定义）
data_sample, label_sample = queue.dequeue()

# 创建包含4个线程的QueueRunner
qr = tf.train.QueueRunner(queue, [enqueue_op] * 4)

with tf.Session() as sess:
    # 创建Coordinator
    coord = tf.train.Coordinator()
    # 启动QueueRunner管理的线程
    enqueue_threads = qr.create_threads(sess, coord=coord, start=True)
    # 主线程，消费100个数据
    for step in range(100):
        if coord.should_stop():
            break
        data_batch, label_batch = sess.run([data_sample, label_sample])
        print (data_batch)
    # 主线程计算完成，停止所有采集数据的进程
    coord.request_stop()
    coord.join(enqueue_threads)

'''







'''

#input1 = tf.placeholder(tf.float32)  #提前站位，分配内存，单是不传入数据
#input2 = tf.placeholder(tf.float32)
# 
#output = tf.multiply(input1, input2)
# 
#with tf.Session() as sess:  #feed_dict的时候才传入数据，并进行计算
#    print(sess.run(output, feed_dict = {input1:[3.], input2: [4.]}))



import pandas as pd

tmp_list = pd.read_csv('fname.csv',header=0)
print (tmp_list)

print (len(tmp_list))
'''





























'''
1.1 画npg格式的数据

import numpy as np
import glob,os
import matplotlib

import matplotlib.pyplot as plt

#画npz数据图，默认的是9001,3 的数据格式，保存的图片的名称是file_name
def plot_npz(file_name,data,itp,its): 
    plt.figure(figsize=(25,15))
    ax=plt.subplot(data.shape[1],1,1) #(3,1,1) 输入的数据shape是9001,3
    if itp=='non':
        for j in range(data.shape[1]):
            plt.subplot(data.shape[1],1,j+1,sharex=ax)
            t=np.linspace(0,data.shape[0]-1,data.shape[0]) #(0,9000,9001)
            plt.plot(t,data[:,j])
    
    else:
        for j in range(data.shape[1]):
            plt.subplot(data.shape[1],1,j+1,sharex=ax)
            t=np.linspace(0,data.shape[0]-1,data.shape[0]) #(0,9000,9001)
            plt.vlines(itp,min(data[:,j]),max(data[:,j]),colors='r')  #画纵轴，有的可能没有
            plt.vlines(its,min(data[:,j]),max(data[:,j]),colors='r')  
            plt.plot(t,data[:,j])
    
    plt.suptitle('test',fontsize=25,color='r')
    png_name=file_name+'.png' 
    plt.savefig(png_name)
    #os.system('mv *.png png') 
    plt.close()



#读取画图PhaseNet-master/dataset/waveform_train目录下的数据(npz格式的数据)
npz_datas = glob.glob('/home/zhangzhipeng/software/github/2020/test/*.npz')
for npz in npz_datas:

    #获得npz数据的文件名，将后缀.npz去掉，作为参数传递给画图子函数。
    file_name = os.path.basename(npz)
    file_name = file_name.replace('.npz','')
    
    #获得npz数据中的data,是一个三分量的数据，作为参数传递给画图子函数。   
    A = np.load(npz)
    ca_names = A.files
    data = A[ca_names[0]]

    #plot_npz(file_name,data,'non','non')

    #获得itp和its的位置，用来在图上画竖线
    itp = A[ca_names[1]]  #是一个数字，比如3001
    its = A[ca_names[2]]
    
    print (itp,its,A[ca_names[3]])
    #
    plot_npz(file_name,data,itp,its)
    print (data[:,1])
    ip   = A[ca_names[3]]
    print (ip)
    #print (data.shape)
    #print (data[:,0].shape)
'''  
  



'''
#1.1写入
#将数据村成npz格式的数据,默认的名字是arr_0,arr_1,arr_2等名字，也可以自定义为c_array
a = np.arange(3)
b = np.arange(4)
c = 12.0  #本来是整性，转换成npz后就变成了numpy
np.savez('test.npz',a,b,c_array=c)
print (a,b,c,type(c))

#1.2 读取
#A = np.load('test.npz')
#显示npz文件中有几个目录
catalog_names = A.files

print (catalog_names)
#print (type(A[catalog_names[2]]))
'''











