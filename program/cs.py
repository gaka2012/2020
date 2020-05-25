#!/usr/bin/python
# -*- coding:UTF-8 -*-

from obspy.core import UTCDateTime
from obspy.core import read
from itertools import groupby
import numpy as np


'''
st = read('/home/zhangzhipeng/software/github/2020/program/sac_data/*.sac')
st.sort(['starttime'])
for tr in st:
    s = tr.stats.sac
    r_time = UTCDateTime(year=s.nzyear,julday=s.nzjday,hour=s.nzhour,minute=s.nzmin,second=s.nzsec,microsecond=s.nzmsec*1000) #参考时刻
    p_time = r_time+s.a
    print (tr.stats.starttime,r_time,p_time,s.a,s.evla,s.dist,s.mag,s.az) #输出绝对时间、参考时间、p到时

'''


import time,os
import subprocess

#subprocess.Popen(['./new','2','2','3'])
#p = subprocess.Popen('./new 2 2 3',shell=True,stdout=subprocess.PIPE)
#out =p.stdout.readlines()
#print ('out == ',out)  


'''
st=read('2019.168.14.52.59.0000.SC.HMS.00.BHZ.D.SAC')
tr=st[0]
co=tr.copy()
#去均值，线性，波形歼灭,然后滤波
co.detrend('demean').detrend('linear').taper(max_percentage=0.05, max_length=10.)
co=co.filter('bandpass',freqmin=1,freqmax=15) #带通滤波
co.filter('highpass',freq=1)#高通滤波
#将滤波后的数据转换成numpy格式
data=np.asarray(co)
'''

'''
st = read('SC.AXI_20180101062906.BHZ.sac')
h0 = st[0].stats.starttime
h1 = st[0].stats.endtime
png_file = 'SC.AXI_20180101062906.BHZ.png'
st.plot(starttime= h0 ,endtime=h1,equal_scale=False,outfile=png_file,color='red',size=(1800,1000)) #单张图，这个size比较好，三张图一起画，默认size就行。
'''

#st[0].detrend('demean').detrend('linear').taper(max_percentage=0.05, max_length=10.)
#st[0]=st[0].filter('bandpass',freqmin=1,freqmax=15) #带通滤波


#png_file1 = 'test.png'
#st.plot(starttime= h0 ,endtime=h1,equal_scale=False,outfile=png_file1,color='red',size=(1800,1000)) #单张图，这个size比较好，三张图一起画，默认size就行。
#st[0].write('SC.AXI_20180101062906.BHZ.sac',format='SAC')

#计算信噪比，输入trace,P波到时，噪声取值长度，p波取值长度，返回计算后的信噪比
def cal_SNR(tr,tp,before_tp,after_tp): 
    print (tr)
    #计算信噪比
    c_st = tr.copy()
    n_st = tr.copy()
    #1 剪切p波前后3秒，并转换为numpy格式数据
    no_data = n_st.trim(tp-before_tp,tp,pad=True, fill_value=0)
    no_data = np.asarray(no_data)
    p_data  = c_st.trim(tp, tp + after_tp,pad=True, fill_value=0)   #默认pad是false,这里是True，当给定的截取时间超出了数据范围会自动补fill_value,如果默认则不画超出的时间。
    p_data  = np.asarray(p_data)
    #2 numpy去绝对值
    ab_data = np.fabs(p_data) #P波取绝对值
    ab_nois = np.fabs(no_data)#噪声取绝对值
    #3 求p波最大值处以噪声均值
    p_max  = max(ab_data) #p波最大值
    n_mean = np.sum(ab_nois)/(len(ab_nois)) #噪声均值
    if n_mean==0:
        snr = -1234
    else:
        snr = round(p_max/n_mean,2)
    return snr

st = read('SC.AXI_20180101062906.BHZ.sac')
tr = st[0]
s = tr.stats.sac
r_time = UTCDateTime(year=s.nzyear,julday=s.nzjday,hour=s.nzhour,minute=s.nzmin,second=s.nzsec,microsecond=s.nzmsec*1000) #参考时刻
p_time = r_time+s.a#p波到时


snr = cal_SNR(tr,p_time,10,3)
print (snr)






