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
filename2 = 'FP_right_list.json'
with open(filename2) as file_obj:
    FP_right = json.load(file_obj)
print (type(FP_right),len(FP_right))

'''


#6.7 读取所有的噪声事件波形数据，生成test.txt用以check_result.py备用。

'''
fa = open('test.txt','a+')

i = 0
datas = glob.glob('/home/zhangzhipeng/software/github/2020/data/noise_data/*.BHZ.sac')
for data in datas:
    #写入数据文件所在路径以及tp到时
    fa.write(data+' '+'-1234')
    fa.write('\n')
    i+=1
fa.close()
print ('there are %s data'%(str(i)))


#根据text.txt中的文件列表，用FP将数据遍历一遍，读取生成的zday1.txt，看其是否是空，空的话说明没有拾取到，对于噪声来说就是正常的，
#将FP拾取到的噪声数据复制到一个位置，画图看一下。

save_picked_noise = '/home/zhangzhipeng/software/github/2020/data/noise_data/FP_pick' 

fa = open('test.txt')
A  = fa.readlines()
fa.close()

total = len(A) #总的噪声的数量
pick_num = 0   #拾取的数量，因为是噪声，拾取说明是错误。


for line in A:
    path,answer = line.split()
    if answer == '-1234':  #说明改事件是个地震，而不是噪声
        try:
            subprocess.call('./picker_func_test %s zday1.txt  522 1206 61 10 7' %(path),shell=True) #得到一个数据的结果，检查zday1.txt中的自动拾取的结果。
        
            fb = open('zday1.txt','r')
            B  = fb.readlines()
            fb.close()
            if len(B)!=0:
                pick_num+=1 
                os.system('cp %s %s'%(path,save_picked_noise))
            os.system('rm zday1.txt')
            
        except Exception as e: #所有异常，输出到文件中
            print (e)
            
print ('there are %s noise, FP picked %s which is wrong'%(total,pick_num))

'''


#遍历生成的噪声数据，画图，看一眼。
'''
data_path  = '/home/zhangzhipeng/software/github/2020/data/noise_data/FP_pick' #遍历一个月的所有数据 
save_png = '/home/zhangzhipeng/software/github/2020/data/noise_data/FP_pick_figure/'  #将sac三分量画图后保存位置。

data_files = sorted(glob.glob(data_path+'/*.BHZ.sac'))       #一个月的所有天数
for data in data_files:
    #读取每一天下的所有sac数据,设置切割的起始时间是16:00:00
    st = read(data) 
    co=st.copy()
    
    #去均值，线性，波形歼灭,然后滤波
    co.detrend('demean').detrend('linear').taper(max_percentage=0.05, max_length=10.)
    co=co.filter('bandpass',freqmin=1,freqmax=15) #带通滤波
    
    save_name = save_png+os.path.basename(data).replace('BHZ.sac','png')
    co.plot(equal_scale=False,outfile=save_name,size=(1800,1000)) 
'''













