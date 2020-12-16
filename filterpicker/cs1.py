#!/usr/bin/python
# -*- coding:UTF-8 -*-

import os,glob,json
import subprocess
from obspy.core import UTCDateTime
import matplotlib.pyplot as plt
from itertools import groupby
import numpy as np
from obspy.core import read





#1.3 读取多个错误拾取的文件夹中的数据，然后画图，对于wrong_pick_data中的数据标记人工拾取与FP拾取的结果，t1是FP拾取的结果，


#data_path  = '/home/zhangzhipeng/software/github/2020/data/no_pick_data/*.BHZ.sac' #遍历一个月的所有数据 
#save_png = '/home/zhangzhipeng/software/github/2020/data/no_pick_figure/'  #将sac三分量画图后保存位置。

#data_path  = '/home/zhangzhipeng/software/github/2020/data/wrong_pick_data/*.BHZ.sac' #遍历一个月的所有数据 
#save_png = '/home/zhangzhipeng/software/github/2020/data/wrong_pick_figure/'  #将sac三分量画图后保存位置。


#data_path  = '/home/zhangzhipeng/software/github/2020/data/wrong_data/*.BHZ.sac' #遍历一个月的所有数据 
#save_png = '/home/zhangzhipeng/software/github/2020/data/wrong_pick_figure/'  #将sac三分量画图后保存位置。


'''
#画npz数据， 数据格式是npz,shape是(12001,)，输入画完图的保存路径‘/’，文件名称，数据
def plot_waveform_npz(plot_dir,file_name,data,itp,its): 
    plt.figure(figsize=(25,15))
    ax=plt.subplot(1,1,1) #(3,1,1) 输入的数据shape是3,9001
    for j in range(1):
        plt.subplot(1,1,j+1,sharex=ax)
        t=np.linspace(0,12000,12001) #(0,2999,3000)
        data_max=data.max()
        data_min=data.min()
        plt.plot(t,data,color = 'black')
        plt.vlines(itp,data_min,data_max,colors='r') 
        plt.vlines(its,data_min,data_max,colors='blue') 
        
    plt.suptitle(file_name,fontsize=25,color='r')
    png_name=file_name+'.png' 
    plt.savefig(plot_dir+png_name)
    #os.system('mv *.png png') 
    plt.close()  


data_files = glob.glob(data_path)       #所有的z分量的数据
for data in data_files:
    name = os.path.basename(data).replace('.BHZ.sac','')
    st = read(data) 
    co=st[0].copy()
    #去均值，线性，波形歼灭,然后滤波
    co.detrend('demean').detrend('linear').taper(max_percentage=0.05, max_length=10.)
    co=co.filter('bandpass',freqmin=1,freqmax=15) #带通滤波
    
    #将滤波后的数据转换成numpy格式，并计算人工与FP拾取的结果，tp是人工拾取，Fp是FP拾取的结果，都画在一起。
    data=np.asarray(co)
    tp = st[0].stats.sac.a
    b  = st[0].stats.sac.b
    Fp = st[0].stats.sac.t1  
    tp_num = int((tp-b)*100) #p波到时的点数是P波到时减去b值，参考时刻是发震时刻，p波到时的点数应该是1500，因为截取的时候是前15s
    FP_num = int((Fp-b)*100)
    
    print (data.shape,tp_num)
    plot_waveform_npz(save_png,name,data,tp_num,FP_num)
'''



'''
fa = open('test.txt','a+')

i = 0
datas = glob.glob('/home/zhangzhipeng/software/github/2020/data/*.BHZ.sac')
for data in datas:
    st = read(data)
    start = st[0].stats.starttime
    at = st[0].stats.sac
    
    
    tp = start+at.a-at.b
    
    #写入数据文件所在路径以及tp到时
    fa.write(data+' '+str(tp))
    fa.write('\n')
    i+=1
fa.close()
print ('there are %s data'%(str(i)))
'''

filename2 = 'FP_right_list.json'
with open(filename2) as file_obj:
    FP_right = json.load(file_obj)


print (type(FP_right),len(FP_right))













































