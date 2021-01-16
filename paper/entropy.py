#!/usr/bin/python
# -*- coding:UTF-8 -*-


import glob,os,json
from obspy.core import read
from datetime import *
from obspy.core import UTCDateTime
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import entropy as sci_entropy

def short_term_energy(chunk):
    return np.sum((np.abs(chunk) ** 2) / chunk.shape[0])

def energy_per_frame(windows):
    out = []
    for row in windows: #row代表每一行
        out.append(short_term_energy(row)) #每个值的平方和再处以5
    return np.hstack(np.asarray(out))
#在一张图中画2个折线图，输入2个y的纵坐标，x根据y的长度自动生成
def plot_line_chart(y,y1):

    #在同一幅图中，画2个折线图
    plt.figure(figsize=(25,15))

    #图一：地震熵值折线图
    ax = plt.subplot(1,1,1)
    #设置xy，
    #y = [2,3,1,2,2,1]   #y轴数据
    x = [i for i in range(len(y))] #根据能量的长度生成x轴。
    #plt.plot(y,marker='o',label='event',color='red')
    plt.scatter(x,y,c='red',s=6,marker='o') #c是颜色，s是画的点的大小
    #在折线图上添加数值
    #text_list = [2.1,3.1,1.1,2.1,2.1,1.1]
    #num=0
    #for x2,y2 in zip(x,y):
    #    plt.text(x2,y2,text_list[num],ha='center',va='bottom',fontsize=10) #在x1,y1位置添加数值
    #    num+=1
    
    #图二：噪声熵值折线图    
    
    #y1 = [3,3.2,3.4,3.5,3.5,3.3]
    #plt.plot(y1,label='noise')
    plt.scatter(x,y1,c='blue',s=6) #c是颜色，s是画的点的大小
    
    name = 'event_entropy_line_chart'
    plt.title(name,fontsize=24,color='r')
    plt.xlabel('x',fontsize=25) #加上横纵坐标轴的描述。
    plt.ylabel('y',fontsize=25)

    #x_show = [0,10,20,30,40,50,60,70,80,90,100]
    x_show = [200,400,600,800,1000,1200]

    y_show = [0,1,2,3,4]
    ax.set_xticks(x_show)             #显示的坐标，只显示0,1,10,其他不显示
    ax.set_xticklabels(x_show,rotation=0,fontsize=30)  #
    ax.set_yticks(y_show)             #显示的坐标，只显示0,1,10,其他不显示
    ax.set_yticklabels(y_show,rotation=0,fontsize=30)  #


    #将图片保存下来。
    plt.legend(loc=0,prop={'size':30})
    plt.savefig('event_noise_entropy')
    plt.close()



#子函数，计算熵值，首先将数据转换成numpy格式，然后作预处理，
def calculate_entropy(tr,tp_num):
    
    preceding_time,duration_time = 10,20 #p波到时前5s，总长度为15s
    noise_time  = 20 #噪声是到时前20s,总长度也是15s
    num = tp_num
    fm = 100 #默认频率是100，秒数乘以频率得到点数。
    
    #转换数据格式
    da   = tr.copy()
    da.detrend('demean').detrend('linear').taper(max_percentage=0.05, max_length=10.)
    da.filter('bandpass',freqmin=1,freqmax=15)
    data  = np.asarray(da)
    
    #计算实际地震熵值
    event = data[num-preceding_time*fm:num-preceding_time*fm+duration_time*fm]  #15秒长的数据    
    event_part   = np.reshape(event,[-1,100])       #reshape一下，将其转换为n(15)个长度为100的片段。
    
    event_energy = energy_per_frame(event_part)  #得到15个片段的能量值
    event_out    = sci_entropy(event_energy) #求熵值,即15个平均能力的熵值。
    
        
    #计算手动拾取第3000个点的熵值
    num = 3000
    noise = data[num-preceding_time*fm:num-preceding_time*fm+duration_time*fm]  #15秒长的数据
    noise_part   = np.reshape(noise,[-1,100])       #reshape一下，将其转换为n(15)个长度为100的片段。
    noise_energy = energy_per_frame(noise_part)  #得到15个片段的能量值
    noise_out    = sci_entropy(noise_energy) #求熵值,即15个平均能力的熵值。

    #return '{:.2f}'.format(round(event_out,2)),'{:.2f}'.format(round(noise_out,2))
    return round(event_out,2),round(noise_out,2)




filename='wrong_dict.json'
with open(filename) as file_obj:
    wrong_dict = json.load(file_obj)


#1.2 读取切割好的长度为90s的sac格式的波形数据，然后计算p波和噪声的熵值。间隔是1s,p波取到时前5s到15s,噪声取到时前5s到前25s,并画熵值分布图
data_files = glob.glob('/home/zhangzhipeng/software/github/2020/data/wrong_pick/*.BHZ.sac')
save_fig   = '/home/zhangzhipeng/software/remos/test/'
event_entropy,noise_entropy = [],[] #存储地震事件和噪声的熵值，用来画图

#遍历所有的数据，得到2个列表，分别存储地震和噪声的熵值
for data_file in data_files:
    data_name = os.path.basename(data_file)
    st = read(data_file)    
    at = st[0].stats.sac
    b,tp,ts = at.b,at.a,at.t0
    if data_name in wrong_dict.keys():
        tp_num = int(wrong_dict[data_name]*100)+3000
        if tp_num<=500:
            tp_num = 500
        try:
            result = calculate_entropy(st[0],tp_num)    
            event_entropy.append(result[0])
            noise_entropy.append(result[1])
        except ValueError:
            print (data_name)


plot_line_chart(event_entropy,noise_entropy)






























