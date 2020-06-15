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

a = '1234main'

b = a.replace('12*','test')

print (help(b.replace))





