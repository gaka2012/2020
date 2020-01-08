#!/usr/bin/python
# -*- coding:UTF-8 -*-

'''
本程序会读取read_report.py生成的result.txt文档，找到拾取震相的地震，然后记录下发震时刻(bj)，计算gmt时刻，记录到时，计算gmt到时，记录震级，可以用于filterpicker对照答案。
'''
import re,os,glob,sys
from obspy.core import UTCDateTime

read_file = 'result.txt'
out_file  = 'mer3.txt'

pha = False
with open(read_file,'r') as fr:
    for line in fr:
        part = line.split() #每一行的开头
        if pha:
            if part[0] == 'DPB':     #说明有拾取的震相
                arrival_time=UTCDateTime(line [33:57]) #拾取的到时
                gmt_arrtime = arrival_time-8*3600
                fa = open(out_file,'a+')
                fa.write(str(start_time)+' '+str(gmt_sttime)+' '+str(arrival_time)+' '+str(gmt_arrtime)+' '+str(magnitude)+'\n')
                fa.close()
                pha = False
            if part[0] == 'DBO':
                start_time = UTCDateTime(part[2]+' '+part[3]) #发震时刻
                gmt_sttime = start_time-8*3600
                magnitude  = abs(float(line[56:61]))          #震级
        else:
            if part[0] == 'DBO': 
                start_time = UTCDateTime(part[2]+' '+part[3]) #发震时刻
                gmt_sttime = start_time-8*3600
                magnitude  = abs(float(line[56:61]))          #震级
                pha        = True
                
