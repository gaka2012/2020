# -*- coding: utf-8 -*-
"""
Created on Tue May  9 16:54:32 2017
@author: 代码医生 qq群：40016981，公众号：xiangyuejiqiren
@blog：http://blog.csdn.net/lijin6249
"""


import cifar10_input
import tensorflow as tf
import numpy as np

#保存和加载模型，就看这个
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
isTrain = False
train_steps = 100  #迭代100次
checkpoint_steps = 50  #每迭代50次就保存一个模型
checkpoint_dir = './test_save/'  #保存模型位置，
saver = tf.train.Saver()  # defaults to saving all variables - in this case w and b
x_data = np.reshape(np.random.rand(10).astype(np.float32), (10, 1))  #生成10个[0,1)区间的随机数，然后reshape一下。


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer()) 
    #判断当前工作状态
    if isTrain: #isTrain:True表示训练；False：表示测试
        # 3.迭代，反复执行上面的最小化损失函数这一操作（train op）,拟合平面
        for i in range(train_steps): #train_steps表示训练的次数，例子中使用100
            sess.run(train, feed_dict={x: x_data})
            if (i + 1) % checkpoint_steps == 0: #表示训练多少次保存一下checkpoints，例子中使用50
                print ('step: {}  train_acc: {}  loss: {}'.format(i, sess.run(w), sess.run(b)))
                saver.save(sess, checkpoint_dir + 'model.ckpt', global_step=i+1) #表示checkpoints文件的保存路径以及名字，
    else: #如果isTrain=False，则进行测试
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        ckpt.model_checkpoint_path = checkpoint_dir + 'model.ckpt'
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path) #恢复变量
        else:
            pass
        print(sess.run(w),sess.run(b))
        

        
        
'''
batch_size = 128
data_dir = './book_data/cifar-10-batches-bin'
print("begin")
images_train, labels_train = cifar10_input.inputs(eval_data = False,data_dir = data_dir, batch_size = batch_size) #训练数据
images_test, labels_test = cifar10_input.inputs(eval_data = True, data_dir = data_dir, batch_size = batch_size)   #测试数据
print("begin data")



def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)  #w的值为截断正态分布，均值默认为0,标准差为0.1,传入shape参数
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)            #b值设为0.1
  return tf.Variable(initial)
  
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')  #卷积，上面的images_train得到的shape是

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')  
                        
def avg_pool_6x6(x):
  return tf.nn.avg_pool(x, ksize=[1, 6, 6, 1],
                        strides=[1, 6, 6, 1], padding='SAME')

# tf Graph Input
x = tf.placeholder(tf.float32, [None, 24,24,3]) # cifar data image of shape 24*24*3
y = tf.placeholder(tf.float32, [None, 10]) # 0-9 数字=> 10 classes


W_conv1 = weight_variable([5, 5, 3, 64])
b_conv1 = bias_variable([64])

x_image = tf.reshape(x, [-1,24,24,3])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 64, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_conv3 = weight_variable([5, 5, 64, 10])
b_conv3 = bias_variable([10])
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)

nt_hpool3=avg_pool_6x6(h_conv3)#10
nt_hpool3_flat = tf.reshape(nt_hpool3, [-1, 10])
y_conv=tf.nn.softmax(nt_hpool3_flat)

cross_entropy = -tf.reduce_sum(y*tf.log(y_conv))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


sess = tf.Session()
sess.run(tf.global_variables_initializer())
tf.train.start_queue_runners(sess=sess)
for i in range(15000):#20000
  image_batch, label_batch = sess.run([images_train, labels_train])
  label_b = np.eye(10,dtype=float)[label_batch] #one hot
  
  train_step.run(feed_dict={x:image_batch, y: label_b},session=sess)
  
  if i%200 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:image_batch, y: label_b},session=sess)
    print( "step %d, training accuracy %g"%(i, train_accuracy))


image_batch, label_batch = sess.run([images_test, labels_test])
label_b = np.eye(10,dtype=float)[label_batch]#one hot
print ("finished！ test accuracy %g"%accuracy.eval(feed_dict={
     x:image_batch, y: label_b},session=sess))
     
'''

