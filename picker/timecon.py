#!/usr/bin/python
# -*- coding:UTF-8 -*-

import glob
from obspy.core import read
from datetime import *
from obspy.core import UTCDateTime


'''
本程序用来将filename中的第6,7,8行的时间转换为UTCdate格式的时间，然后减去be_time时间，得到相差的秒数。
'''


filename='zorigin.txt' #读取的文件名 （格式为 20190617 2011 12.67） 
data_column = 6          #日期在第几列,从0开始
hour        = 7          #小时在第几列
second      = 8          #秒数在第几列
be_time     = '2019-06-17T16:00:00.070' #开始的时间



be_time=UTCDateTime(be_time)
fr=open(filename,'r')
aline=fr.readlines()
fr.close()
for line in aline:
    part=line.split()
    s1,s2,s3=part[6],part[7],part[8]
    newtime=UTCDateTime(s1+' '+s2+' '+s3)
    subtime=newtime-be_time  
    print (subtime)
#time=UTCDateTime('20190618' '2011' '12.67')
#print (time)
