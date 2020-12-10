#!/usr/bin/python
# -*- coding:UTF-8 -*-

import os,glob
import subprocess
from obspy.core import UTCDateTime
import matplotlib.pyplot as plt
from itertools import groupby
import numpy as np
from obspy.core import read


'''
fa = open('test.txt','a+')

datas = glob.glob('/home/zhangzhipeng/software/github/2020/data/*.BHZ.sac')
i=0
for data in datas:
    st = read(data)
    start = st[0].stats.starttime
    at = st[0].stats.sac
    
    
    tp = start+at.a-at.b
    
    #print (start,at.b,tp)
    #print (at.b)
    #print (st[0].stats.starttime)
    
    fa.write(data+' '+str(tp))
    fa.write('\n')
    i+=1
fa.close()
print (i)
'''



#/home/zhangzhipeng/software/github/2020/data/SC.AXI_20180413035538.BHZ.sac 2016-03-13T04:53:26.810000Z

#1.3 sac数据画图

#data_path  = '/home/zhangzhipeng/software/github/2020/data/no_pick_data/*.BHZ.sac' #遍历一个月的所有数据 
#save_png = '/home/zhangzhipeng/software/github/2020/data/no_pick_figure/'  #将sac三分量画图后保存位置。

data_path  = '/home/zhangzhipeng/software/github/2020/data/wrong_pick_data/*.BHZ.sac' #遍历一个月的所有数据 
save_png = '/home/zhangzhipeng/software/github/2020/data/wrong_pick_figure/'  #将sac三分量画图后保存位置。



data_files = glob.glob(data_path)       #一个月的所有天数
for data in data_files:
    st = read(data) 
    co=st[0].copy()
    #去均值，线性，波形歼灭,然后滤波
    co.detrend('demean').detrend('linear').taper(max_percentage=0.05, max_length=10.)
    co=co.filter('bandpass',freqmin=1,freqmax=15) #带通滤波
    
    data_name = os.path.basename(data)
    save_name =save_png+data_name.replace('sac','png')
    co.plot(outfile=save_name,size=(1800,1000)) 


















