#!/usr/bin/env python
#-*- coding:utf-8 -*-

import os,sys,glob,re,csv,json
import numpy as np
from obspy.core import read
import matplotlib.pyplot as plt
from itertools import groupby
#1.1读取震相报告获得人工拾取字典，
#加载字典,里面是AXI台站一个月2018/01的震相报告，不管发震时刻，只把p波到时提取出来，一个月才94个地震。

'''
filename='/home/zhangzhipeng/temp_delete/info_dict.json'
with open(filename) as file_obj:
    p_s_dict = json.load(file_obj)
    
   
for key,value in p_s_dict.items():
    print (key,len(value)) 

'''



#1.2 读取切割好的长度为120s的sac格式的波形数据，然后画图
'''
data_files = glob.glob('/home/zhangzhipeng/software/github/2020/data/*.BHZ.sac')
save_fig   = '/home/zhangzhipeng/software/github/2020/figure/'

total = len(data_files)
num = 0

for data_file in data_files:
    three_c = data_file.replace('BHZ.sac','*')
    st = read(three_c)
    st.sort(keys=['channel'], reverse=False) #对三分量数据排序
    
    at = st[0].stats.sac
    b,tp,ts = at.b,at.a,at.t0
        
    tp_num = int((tp-b)*100) #p波到时的点数是P波到时减去b值，参考时刻是发震时刻，p波到时的点数应该是1500，因为截取的时候是前15s
    ts_num = int((ts-b)*100) #s波到时有负数，比如-1234，再乘以100
    
    png_file = os.path.basename(data_file).replace('BHZ.sac','png')
    png_file = save_fig+png_file
    st.plot(equal_scale=False,outfile=png_file,color='red',size=(1800,1000)) #单张图，这个size比较好，三
    #print (tp_num,ts_num)

    num+=1
    percent=num/total #用于写进度条
    sys.stdout.write("\r{0}{1}".format("#"*10 , '%.2f%%' % (percent * 100)))
    sys.stdout.flush()
'''

#1.3 读取截取好的地震事件，画震级分布图
def plot_num_distribution(num_list,fig_name): #画不同震级的数量分布图.输入不同的震级形成的列表,处数，震级最大值,横纵坐标轴名
    
    #统计num_list中每个震级的数量,形成2个列表，一个震级，一个是震级对应的数量
    x_list,y_list = [],[] 
    for mag,num in groupby(sorted(num_list),key=lambda x:x): 
        print (mag)
        x_list.append(mag)
        y_list.append(len(list(num)))
    #x,y坐标轴显示的坐标
    new_x = [0,1,2,3,4,5,6,7,8]
    new_y = [0,20,40,60,80,100]
    
    #画图
    plt.figure(figsize=(25, 15))
    plt.bar(x_list,y_list,color='gray',width=0.1,ec='black') #柱状图的宽度,描边
    plt.xticks(new_x,fontsize=30)
    plt.yticks(new_y,fontsize=30)
    plt.xlabel('mag Ms',fontsize=30) #加上横纵坐标轴的描述。
    plt.ylabel('count',fontsize=30) #
    
    plt.savefig(fig_name)  #注意要在plt.show之前
    #plt.close()

data_files = glob.glob('/home/zhangzhipeng/software/github/2020/data/*.BHZ.sac')


num_list = [] 
for data_file in data_files:
    st = read(data_file)
    at = st[0].stats.sac
    
    mag = at.mag
    num_list.append(mag)
    
print (num_list)
#for mag,num in groupby(sorted(num_list),key=lambda x:x): 
#    print (mag)

plot_num_distribution(num_list,'mag_dis.png')



'''
y_list = [1,2,3,4,5,6,7] #每个震级对应的地震数量
x_list = [1.1,1.2,1.3,1.4,2,2.1,2.2]           #震级分布
new_x  = [0,1,2,3]  #横轴只显示这些坐标
xlab,ylab = 'mag Ms','count'

#plt.bar(x_list,y_list,color='red',tick_label=name_list,width=0.1) #柱状图的宽度
plt.bar(x_list,y_list,color='gray',width=0.1,ec='black') #柱状图的宽度,描边
plt.xticks(new_x,fontsize=30)
plt.yticks(new_y,fontsize=30)
plt.xlabel(xlab,fontsize=30) #加上横纵坐标轴的描述。
plt.ylabel(ylab,fontsize=30) #
plt.show()
plt.close()
'''










