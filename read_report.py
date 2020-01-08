#!/usr/bin/python
# -*- coding:UTF-8 -*-
'''
python3 read_report.py -t 2015-07-01T12:00:00 2015-08-10T20:00:00 -lat 3 50 -lon 70 150 -m 4,10 -sta BAC -pha P,Pg,Pn,S,Sg
必须有的选项 -t
可选选项    -lat -lon -m -sta -pha

'''

import re,os,glob,sys
from obspy.core import UTCDateTime
import matplotlib.pyplot as plt
import numpy as np

in_put   = sys.argv

report_path = 'report'       #震相报告所在路径
out_name    = 'result.txt'
mag_index   = False
sta_index   = False
phase_index = False
lat_index   = False
lon_index   = False
for content in in_put:
    if content == '-t':
        start_time_index = in_put.index(content)
    elif content == '-m':
        mag_index = in_put.index(content)
    elif content == '-lat':
        lat_index = in_put.index(content)
    elif content == '-lon':
        lon_index = in_put.index(content)
    elif content == '-sta':
        sta_index = in_put.index(content)
    elif content == '-pha':
        phase_index = in_put.index(content)
start_time = in_put[start_time_index+1]
end_time   = in_put[start_time_index+2]
begin_time = UTCDateTime(start_time)
end_time   = UTCDateTime(end_time)


if mag_index: #震级，默认是0-10
    mag  = in_put[mag_index+1].split(',')
    mag_min,mag_max = float(mag[0]),float(mag[1])
    #print (mag_min,mag_max)
else:
    mag_min,mag_max=0,10
if sta_index:
    sta_list = in_put[sta_index+1].split(',')
    #print (sta_list)
if phase_index:
    phase_list = in_put[phase_index+1].split(',')
    #print (phase_list)
if lat_index:
    lat_min    = float(in_put[lat_index+1])
    lat_max    = float(in_put[lat_index+2])
else:
    lat_min    = -90
    lat_max    = 90
if lon_index:
    lon_min    = float(in_put[lon_index+1])
    lon_max    = float(in_put[lon_index+2])
else:
    lon_min,lon_max = -180,180
#print (mag_min,mag_max)
earth_num=0 #震相报告中的地震数量
phase_num=0 #最终挑选的震相的数量
earth=False
report_files= glob.glob(report_path+'/*')
total=len(report_files)
num=0
for report_file in report_files:
   line_num=0
   num+=1
   percent=num/total
   sys.stdout.write("\r{0}{1}".format("#"*10 , '%.2f%%' % (percent * 100)))
   sys.stdout.flush()
   with open(report_file,'r',encoding='gbk') as fr:
       for line in fr:
           i=0
           line_num+=1
           part=line.split()
           try:
               if part[0]=='DBO' : #DBO表示是某个地震
                   bjtime  = UTCDateTime(part[2]+' '+part[3]) #发震时刻
                   magnitude = abs(float(line[56:61]))             #震级
                   if lat_min<=float(part[4])<=lat_max and  lon_min<=float(part[5])<=lon_max and  begin_time<=bjtime<=end_time and mag_min<=magnitude<=mag_max:
                       #print (magnitude)
                       earth=True #符合条件才会继续寻找相应的拾取台站到时信息
                       gmttime = bjtime-8*3600
                       earth_num+=1
                       fa=open(out_name,'a+')
                       fa.write(line)
                       fa.close()
                   else :
                       earth=False
               elif earth:
                   if not sta_index and not phase_index: #如果没有提交台站以及震相
                       if part[0] == 'DPB':              #表明是台站报告，而不是其他乱七八糟的
                           fa=open(out_name,'a+')
                           fa.write(line)
                           fa.close()
                   elif sta_index and not phase_index:   #如果提交了台站没有提交震相
                       if part[2] in sta_list:
                           fa=open(out_name,'a+')
                           fa.write(line)
                           fa.close()
                   elif sta_index and phase_index :      #如果提交了台站和震相
                       if part[2] in sta_list and (line[24:28].split()[0] in phase_list):
                           fa=open(out_name,'a+')
                           fa.write(line)
                           fa.close()
           except Exception as e:
               continue
               #print (e,report_file,line_num)
print ()
print ('total earthquake ==',earth_num)
#print ('total phase      ==',phase_num)











