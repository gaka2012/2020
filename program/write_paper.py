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


#1.4读取phasenet生成的结果.csv，读取每个拾取，不用关心文件名，只统计第2列，即p波拾取，如果时空，则miss_num加一，如果与1500的差值大于等于100，则big_100加一
out_name  = '/home/zhangzhipeng/software/github/2020/output/phasenet/picks.csv' #存储结果的csv文件
#1.1.1先获取原始数据中的所有文件名称
miss_num = 0  #下面的if中，如果row[1]是空，说明没有拾取到地震事件，则miss_num加一
big_100  = 0  #如果拾取到了(可能有2个拾取，只取第一个拾取)，计算其与1500的差值，如果差值的绝对值大于等于100，说明误差大于1s，则计数加一
with open(out_name) as f:
    reader = csv.reader(f)
    header = next(reader)
    for row in reader:
        name,tp = row[0],row[1]
        if row[1]=='[]':
            miss_num+=1
        else:
            pick_nums = re.findall('\d+',row[1]) #row[1]是str--'[1448]'，找到其中的数字，并转换成整数，有的有2个拾取
            pick_num = int(pick_nums[0])
            if abs(pick_num-1500) >= 100:
                big_100+=1
print (miss_num,big_100) 


#1.5 读取切割好的长度为120s的sac格式的波形数据，然后计算p波和噪声的熵值。间隔是1s,p波取到时前5s到15s,噪声取到时前5s到前25s,并画熵值分布图        


'''
import glob,remos,os
from obspy.core import read
from datetime import *
from obspy.core import UTCDateTime
import numpy as np
import matplotlib.pyplot as plt
import newfilter
from scipy import signal
import segment_axis as sa
from scipy.stats import entropy as sci_entropy


#在一张图中画2个折线图，输入2个y的纵坐标，x根据y的长度自动生成
def plot_line_chart(y,y1):

    #在同一幅图中，画2个折线图
    plt.figure(figsize=(25,15))

    #图一：地震熵值折线图
    ax = plt.subplot(1,1,1)
    #设置xy，
    #y = [2,3,1,2,2,1]   #y轴数据
    x = [i for i in range(len(y))] #根据能量的长度生成x轴。
    plt.plot(y,marker='o',label='event',color='red')
    
    #在折线图上添加数值
    #text_list = [2.1,3.1,1.1,2.1,2.1,1.1]
    #num=0
    #for x2,y2 in zip(x,y):
    #    plt.text(x2,y2,text_list[num],ha='center',va='bottom',fontsize=10) #在x1,y1位置添加数值
    #    num+=1
    
    #图二：噪声熵值折线图    
    
    #y1 = [3,3.2,3.4,3.5,3.5,3.3]
    plt.plot(y1,label='noise')
    
    name = 'event_entropy_line_chart'
    plt.title(name,fontsize=24,color='r')
    plt.xlabel('x',fontsize=25) #加上横纵坐标轴的描述。
    plt.ylabel('y',fontsize=25)

    x_show = [1,2,3,4]
    y_show = [0,1,2,3,4]
    ax.set_xticks(x_show)             #显示的坐标，只显示0,1,10,其他不显示
    ax.set_xticklabels(x_show,rotation=0,fontsize=30)  #
    ax.set_yticks(y_show)             #显示的坐标，只显示0,1,10,其他不显示
    ax.set_yticklabels(y_show,rotation=0,fontsize=30)  #


    #将图片保存下来。
    plt.legend(loc = 0, prop = {'size':30}) #加上这个就会显示出来。
    plt.savefig('event_noise_entropy')
    plt.close()



#子函数，计算熵值，首先将数据转换成numpy格式，然后作预处理，
def calculate_entropy(tr,tp_num):
    
    preceding_time,duration_time = 5,15 #p波到时前5s，总长度为15s
    noise_time  = 20 #噪声是到时前20s,总长度也是15s
    num = tp_num
    fm = 100 #默认频率是100，秒数乘以频率得到点数。
    
    #转换数据格式
    da   = tr.copy()
    da.detrend('demean').detrend('linear').taper(max_percentage=0.05, max_length=10.)
    da.filter('bandpass',freqmin=1,freqmax=15)
    data  = np.asarray(da)
    
    #计算地震熵值
    event = data[num-preceding_time*fm:num-preceding_time*fm+duration_time*fm]  #15秒长的数据
    event_part   = np.reshape(event,[-1,100])       #reshape一下，将其转换为n(15)个长度为100的片段。
    event_energy = remos.energy_per_frame(event_part)  #得到15个片段的能量值
    event_out    = sci_entropy(event_energy) #求熵值,即15个平均能力的熵值。
    
        
    #计算噪声熵值
    noise = data[num-noise_time*fm:num-noise_time*fm+duration_time*fm]  #15秒长的数据
    noise_part   = np.reshape(noise,[-1,100])       #reshape一下，将其转换为n(15)个长度为100的片段。
    noise_energy = remos.energy_per_frame(noise_part)  #得到15个片段的能量值
    noise_out    = sci_entropy(noise_energy) #求熵值,即15个平均能力的熵值。

    #return '{:.2f}'.format(round(event_out,2)),'{:.2f}'.format(round(noise_out,2))
    return round(event_out,2),round(noise_out,2)


#1.2 读取切割好的长度为120s的sac格式的波形数据，然后计算p波和噪声的熵值。间隔是1s,p波取到时前5s到15s,噪声取到时前5s到前25s,并画熵值分布图
data_files = glob.glob('/home/zhangzhipeng/software/github/2020/data/*.BHZ.sac')
save_fig   = '/home/zhangzhipeng/software/remos/test/'
event_entropy,noise_entropy = [],[] #存储地震事件和噪声的熵值，用来画图

#遍历所有的数据，得到2个列表，分别存储地震和噪声的熵值
for data_file in data_files:
    st = read(data_file)    
    at = st[0].stats.sac
    b,tp,ts = at.b,at.a,at.t0
    tp_num = int((tp-b)*100) #p波到时的点数是P波到时减去b值，参考时刻是发震时刻，p波到时的点数应该是1500，因为截取的时候是前15s
    
    result = calculate_entropy(st[0],tp_num)    
    event_entropy.append(result[0])
    noise_entropy.append(result[1])
    

print (event_entropy,noise_entropy)    

plot_line_chart(event_entropy,noise_entropy)

'''



