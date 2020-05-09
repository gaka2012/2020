#!/usr/bin/python
# -*- coding:UTF-8 -*-
'''
python3 read_report.py -t 2015-07-01T12:00:00 2015-08-10T20:00:00 -lat 3 50 -lon 70 150 -m 4,10 -sta SC,AXI;BJ,BAC -pha P,Pg,Pn,S,Sg
必须有的选项 -t
可选选项    -lat -lon -m -sta -pha
-sta选项 台网，台站;台网，台站
生成的结果保存在result.txt中。
部分地震事件无经纬度或无震级，会被过滤。
生成月份列表时，会在close_time的基础上加一个月，所以可能会出现没有当前月份文件的情况，需要在report文件中加上一个空白月份。
'''

import re,os,glob,sys
from obspy.core import UTCDateTime
import matplotlib.pyplot as plt
import numpy as np
from datetime import *
import pandas as pd

in_put   = sys.argv

report_path= '/home/zhangzhipeng/software/github/2020/report'       #震相报告所在路径
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
close_time   = in_put[start_time_index+2]
begin_time = UTCDateTime(start_time)
end_time   = UTCDateTime(close_time)

#根据输入的起始和结束时间生成报告列表['2012-10.txt', '2012-11.txt']
close_time = datetime.strftime(datetime.strptime(close_time,"%Y-%m-%dT%H:%M:%S")+timedelta(days=31),'%Y-%m-%dT%H:%M:%S') #当前月份加一个月 2012-11-02T00:00:00
df = pd.date_range(start = start_time.split('T')[0], end = close_time.split('T')[0], freq='M') #按月迭代，注意start和end中必须是年月日，迭代后的是每个月的月底日期，但是缺少最后一个月的。
df2= df.strftime('%Y-%m') #整理成格式 2017-08(str)
file_list = [i+'.txt' for i in df2]

if mag_index: #震级，默认是0-10
    mag  = in_put[mag_index+1].split(',')
    mag_min,mag_max = float(mag[0]),float(mag[1])
    #print (mag_min,mag_max)
else:
    mag_min,mag_max=0,10

#形成台网：台站字典    
if sta_index:
    sta_list = in_put[sta_index+1].split(';') #['SC,AXI,BXI', 'BJ,CD1,CD2']
    net = {}
    for name in sta_list:
        n_list = name.split(',')
        net[n_list[0]] = n_list[1:] #{'SC': [], 'BJ': ['CD1', 'CD2']}

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
total_earth_num=0 #震相报告中的地震数量
end_ear = 0       #最终挑选的地震数量
phase_num=0 #最终挑选的震相的数量
earth=False

#total=len(report_files)
num=0
num_wrong = 0


for report_file in file_list:
   report_file = os.path.join(report_path,report_file)
   line_num=0
   num+=1
   #percent=num/total
   #sys.stdout.write("\r{0}{1}".format("#"*10 , '%.2f%%' % (percent * 100)))
   #sys.stdout.flush()
   with open(report_file,'r',encoding='gbk') as fr:
       for line in fr:
           i=0
           line_num+=1
           part=line.split()
           try:
               if part[0]=='DBO' : #DBO表示是某个地震
                   
                   bjtime  = UTCDateTime(part[2]+' '+part[3]) #发震时刻
                   
                   #如果时间符合则认为该地震ok
                   if begin_time<=bjtime<=end_time:
                       total_earth_num+=1 #所有地震数量
                   
                       #震级有时没有,设置为-12345
                       try:
                           magnitude = abs(float(line[56:61]))             #
                       except ValueError:
                           magnitude = -12345
                       #经纬度有时没有，设置为-12345 
                       try:
                           lat = float(part[4])
                           lon = float(part[5]) 
                       except ValueError:
                           lat = -12345
                           lon = -12345             
                               
                       if lat_min<=lat<=lat_max and  lon_min<=lon<=lon_max and mag_min<=magnitude<=mag_max:
                       #print ('2==',magnitude)
                           earth=True #符合条件才会继续寻找相应的拾取台站到时信息
                           end_ear+=1
                           gmttime = bjtime-8*3600
                           fa=open(out_name,'a+')
                           fa.write(line)
                           fa.close()
                       else :
                           num_wrong+=1
                           earth=False
               elif earth:
                   if not sta_index and not phase_index: #如果没有提交台站以及震相,则把所有的DPB开头的都写入到文件。
                       if part[0] == 'DPB':  
                           fa=open(out_name,'a+')
                           fa.write(line)
                           fa.close()
                   elif sta_index and not phase_index:   #如果提交了台站没有提交震相
                       for key,value in net.items():
                           if len(value)==0:
                               if part[1] == key:
                                   fa=open(out_name,'a+')
                                   fa.write(line)
                                   fa.close()
                           else:
                               if part[1] == key and part[2] in value: 
                                   fa=open(out_name,'a+')
                                   fa.write(line)
                                   fa.close()
                                   
                   elif sta_index and phase_index :      #如果提交了台站和震相
                       for key,value in net.items():
                           if len(value)==0 :
                               if part[1] == key and (line[24:28].split()[0] in phase_list):
                                   fa=open(out_name,'a+')
                                   fa.write(line)
                                   fa.close()
                           else:
                               if part[1] == key and part[2] in value and (line[24:28].split()[0] in phase_list): 
                                   fa=open(out_name,'a+')
                                   fa.write(line)
                                   fa.close()

           except Exception as e:
               continue
               print (e,report_file,line_num)
print ()
print ('total earthquake ==',total_earth_num)
print ('get earthquake   ==',end_ear)
print ('earthquake don not meet the condition ==',num_wrong)
#print ('total phase      ==',phase_num)
#arrival_time= line [33:57] #拾取的到时









