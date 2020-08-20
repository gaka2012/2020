#!/usr/bin/python
# -*- coding:UTF-8 -*-

import os
import subprocess
from obspy.core import UTCDateTime
import matplotlib.pyplot as plt
from itertools import groupby
import numpy as np


def plot_bar(right_list,left_list): #画柱状图，先把左右图的数据准备好。
    #right_list = [27, 7, 4, 2, 2, 1, 3, 0, 0, 1, 0, 0, 2, 1] #右半部分y轴数据
    a = [0+i*0.2 for i in range(14)]  #横坐标是0-13,大于13的不要了
    x = np.array(a)  #x轴坐标，间隔是0.2

    #left_list = [7, 1, 1, 1, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0] #左半部分y轴数据
    b = [0-i*0.2 for i in range(14)]
    x1= np.array(b)

    x_label = [-2,-1,0,1,2] #在哪个位置写x轴标签
    y_label = [10,20]

    width=0.2
    fig,ax = plt.subplots(figsize=(25, 15), dpi=100) #设置像素
    test1 = ax.bar(x+width/2,right_list,width,color='lightgreen',edgecolor='black') #画2遍，第一遍画右半部分的图像，默认是画在横坐标的中间，加上width/2后就画在了靠右边一点。
    test2 = ax.bar(x1-width/2,left_list,width,color='lightgreen',edgecolor='black')#第二遍画左边的图像。
    #plt.bar(x,num_list,color='red',tick_label=name_list,width=0.1) #柱状图的宽度
    plt.xticks(x_label,('-1','-0.5','0','0.5','1'))  #写x轴标签
    plt.yticks(y_label)
    plt.tick_params(labelsize=15) #设置xy轴的字体大小。
    plt.xlabel('time residual (s)',fontsize=15)
    plt.ylabel('number of picks',fontsize=15)
    plt.savefig('test2.png')
    plt.show()
    plt.close()

def add_zero(a):#检查列表，正常应该是0-13每个数字对应一个number，对于没有number的数字设为0,大于13的予以剔除。   
    y = [] #画图时存储y值。
    count = 0
    i = 0
    for num in range(14):  #检查列表，正常应该是0-13每个数字对应一个number，对于没有number的数字设为0,大于13的予以剔除。   
        if count==14:
            break
        elif a[i][0]==count:
            y.append(a[i][1])
        elif a[i][0]!=count:
            y.append(0)
            i-=1
        count+=1
        i+=1
    return y
    
#输入一个列表，里面是元组，取元组的第一个数，如果是正的，添加到pos列表中，如果是负的，添加到neg中，然后对2个列表进行排序以及分割，间距是0.1,需要调用add_zero函数。
#最后返回的是已经处理好的可以用于画图的列表，列表中的数据间隔是0.1(1)
def sta_list(tuple_list):
    pos,neg = [],[] #正数和负数分成2个列表
    right,left = [],[] #统计正数间隔为0.1的个数，以及负的间隔为0.1的个数
    
    for i in tuple_list:
        if i[0]>=0:
            pos.append(i[0])
        else:
            neg.append(i[0])
            
    for k,g in groupby(sorted(pos),key=lambda x:x*10//1): #统计正数中每个数字的个数。
        #print (k,len(list(g)))
        tup = (k,len(list(g))) #0.0代表范围是0-0.1,  1.0代表范围是0.1-0.2
        right.append(tup)
        
    for k,g in groupby(sorted(neg),key=lambda x:x*(-10)//1): #统计负数中每个数字的个数。
        #print (k,len(list(g)))
        tup = (k*(-1),len(list(g))) #0.0代表范围是0-0.1,  1.0代表范围是0.1-0.2
        left.append(tup)
    
    #最终得到的right1和left1是经过补零的从0-13的横坐标对应的纵坐标
    right1 = add_zero(right)
    left = sorted([(i[0]*-1,i[1]) for i in left],key=lambda x:x[0])   #将列表中的负数乘以负1,并重新排序
    left1 = add_zero(left)
    return right1,left1
    
    
#画散点图，输入一个列表，列表中的元素是一个元组，里面是y轴值以及需要在y轴上显示的文字，输入一个最大值，限制了y轴最大值不能超过这个，否则会被设为最大值。
def plot_scatter(yz_list,max_y):
    y,z = [],[]  #纵坐标的数值与在纵坐标中显示的数字
    for ele in yz_list:
        if ele[0]>max_y:
            y.append(max_y)
            z.append(ele[1])
        else:
            y.append(ele[0])
            z.append(ele[1])
    x = [i for i in range(len(y))] #根据纵坐标生成横坐标
    print (y)

    plt.figure(figsize=(25,15))
    #画图
    plt.scatter(x,y,c='g',s=20,label='1') #c是颜色，s是画的点的大小

    #在折线图上加上数字
#    i=0
#    for a1,b1 in zip(x,y): #前面2个参数是位置，
#        plt.text(a1,b1+0.2,z[i],ha='center',va='bottom',fontsize=12)   #plt.text(1,1,c,ha='center',va='bottom',fontsize=7)
#        i+=1

    #横纵轴坐标的描述
    plt.xlabel('x',fontsize=25)
    plt.ylabel('y',fontsize=25)

    plt.savefig('test1')
    #plt.show()    
    plt.close()


#读取filterpicker生成的结果文件中的时间(zday1.txt)，将其中的6,7,8列转换为UTC时间格式。
#输入filterpicker生成的结果文件名称，输入标准答案，计算人工拾取与自动拾取的时间差，找到绝对值最小的值，以及其所在的位置。
def read_result(filename,man_made):
    fr=open(filename,'r')
    aline=fr.readlines()
    fr.close()
    min_sub = 100  #时间差最小值
    i       = 0    #行数
    min_i   = 0    #时间差最小所在的行数。
    for line in aline:
        i+=1
        part=line.split()
        s1,s2,s3 = part[6],part[7],part[8]
        newtime  = UTCDateTime(s1+' '+s2+' '+s3) #自动拾取的时间
        subtract = newtime-man_made              #自动与手动拾取的时间差。
        #找到时间差最小的
        if abs(subtract) < abs(min_sub):
            min_sub = subtract
            min_i   = i
    return min_sub,min_i #返回的是时间差的绝对值最小值，但是返回的不是绝对值，有正有负。
            
            



#1 第一步：读取test.txt中的数据路径和答案,存储到A中，将数据路径赋值给要调用的程序，得到结果，与标准答案进行对比。
fa = open('test.txt')
A  = fa.readlines()
fa.close()


result = []  #将最好的时间差记下来，里面的内容应该是元组形式的
for line in A:
    path,answer = line.split()
    if answer != '-1234':  #说明改事件是个地震，而不是噪声
        subprocess.call('./picker_func_test %s zday1.txt  522 1206 61 10 7' %(path),shell=True) #得到一个数据的结果，检查zday1.txt中的自动拾取的结果。
        
        #计算人工拾取与自动拾取的差
        man_result  = UTCDateTime(answer)  #人工拾取
        min_result  = read_result('zday1.txt',man_result) #调用函数计算自动拾取与手动拾取的误差最小值
        result.append(min_result)
        os.system('rm zday1.txt')
        #print (answer)

#print (len(result),result)  #70个地震自动与手动的时间差,以及是第几个地震的时间差最小
#plot_scatter(result,2)      #画散点图，自己看的

right,left = sta_list(result)  #处理统计时间差，将其整理好，以备画图时用。
plot_bar(right,left)

















