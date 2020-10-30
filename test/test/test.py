#!/usr/bin/python
# -*- coding:UTF-8 -*-

import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


import tensorflow as tf
import numpy as np
import pandas as pd
import threading,time 




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

#之后后缀表示保存的类型，前面的model.ckpt是默认的名称，也可以更改。
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
checkpoint_dir = './'     #当前路径
saver = tf.train.Saver()  # defaults to saving all variables - in this case w and b
x_data = np.reshape(np.random.rand(10).astype(np.float32), (10, 1))  #生成10个[0,1)区间的随机数，然后reshape一下。
 
############################################################
# 以下是用 tf 来解决上面的任务

 
# 设置tensorflow对GPU的使用按需分配
#config  = tf.ConfigProto()
#config.gpu_options.allow_growth = True
 

# 2.启动图 (graph)
with tf.Session() as sess:

    summary_writer = tf.summary.FileWriter('./log',sess.graph)
    
    sess.run(tf.initialize_all_variables()) 
    #判断当前工作状态
    if isTrain: #isTrain:True表示训练；False：表示测试
        # 3.迭代，反复执行上面的最小化损失函数这一操作（train op）,拟合平面
        for i in range(train_steps): #train_steps表示训练的次数，例子中使用100
            sess.run(train, feed_dict={x: x_data})
            if (i + 1) % checkpoint_steps == 0: #表示训练多少次保存一下checkpoints，例子中使用50
                print ('step: {}  train_w: {}  train_b: {}'.format(i, sess.run(w), sess.run(b)))
                saver.save(sess, checkpoint_dir + 'model.ckpt', global_step=i+1) #sess表示保存当前图的结果，表示checkpoints文件的保存路径，例子中使用当前路径
                                                                                 #global_step,可选，编号checkout名字
    else: #如果isTrain=False，则进行测试
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path) #恢复变量
        else:
            pass
        print(sess.run(w),sess.run(b))


'''
#加载模型
with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)  #其实是读取checkpoint文件,参数是文件所在路径，print后发现是checkpoint文件中的内容
    print (ckpt)                                               
    #saver.restore(sess, ckpt.model_checkpoint_path) #读取文件中的第一个模型
    saver.restore(sess, './model.ckpt-50')           #文件中有不同模型的具体路径
    print(sess.run(w),sess.run(b))
'''


'''
#配置session:GPU和CPU
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

enqueue = queue.enqueue([b,b1])
dequeue = queue.dequeue_many(1)

with tf.Session() as sess:
    sess.run(enqueue)
    print (sess.run(dequeue))

'''
 
 
'''
with tf.Session() as sess:
    q = tf.FIFOQueue(3, 'float')              #创建一个长度为3的队列，数据类型为float.
    init = q.enqueue_many(([0.1, 0.2, 0.3],)) #将零个或多个元素编入队列之中
    init2 = q.dequeue()                       #把元素从队列中移出，如果在执行该操作时队列已空，那么将会阻塞直到元素出列，返回出列的tensors的tuple
    init3 = q.enqueue(4.)                     #将一个元素编入该队列，如果在执行该操作时队列已满，那么将会阻塞
 
    sess.run(init)    
    sess.run(init2)
    sess.run(init3)
 
    quelen = sess.run(q.size())  #不要被前面的sess.run迷惑了，不用管它，就是一个执行后面命令的指示，实际上要表达的只是q.size而已。
    for i in range(quelen):
        print(sess.run(q.dequeue()))

'''







#input1 = tf.placeholder(tf.float32)  #提前站位，分配内存，单是不传入数据
#input2 = tf.placeholder(tf.float32)
# 
#output = tf.multiply(input1, input2)
# 
#with tf.Session() as sess:  #feed_dict的时候才传入数据，并进行计算
#    print(sess.run(output, feed_dict = {input1:[3.], input2: [4.]}))



'''
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











