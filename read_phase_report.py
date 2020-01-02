#!/usr/bin/python
# -*- coding:UTF-8 -*-

#1 以下程序会读取report_path路径下所有的震相报告，提取其中符合时间以及经纬度的相应台站的震中距和走时，存放到out_path中 
#  命名规则是台站-震相.dat
#2 读取out_path中与st对应的台站走时表，绘制走时表图，每个台站一张。存放到plot_path中。
# Pg:红  Pn:黑  Sg：蓝 Sn:绿
import re,os,glob,sys
from obspy.core import UTCDateTime
import matplotlib.pyplot as plt
import numpy as np

sta_list=['WDT','ZHQ','JZG','QCH','PWU','JMG','ZJG','AXI','WCH','CD2','YZP','BAX','MDS','TQU','GZA','HMS']
#st          = ['HWS','ANZ']         #需要寻找的台站
phase_list  = ['Pg','Sg','Pn','Sn','P','S'] #需要寻找的震相
report_path = 'report'       #震相报告所在路径
out_name    = 'result.txt'                   #将结果文件移动到此路径下
lat_min     = 29
lat_max     = 34
lon_min     = 101
lon_max     = 107
begin_time  = UTCDateTime('2017-01-01 00:00:00')
end_time    = UTCDateTime('2019-12-31 00:00:00')


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
                   magnitude = float(line[56:62])             #震级
                   if lat_min<=float(part[4])<=lat_max and  lon_min<=float(part[5])<=lon_max and  begin_time<=bjtime<=end_time and mag_min<=magnitude<=mag_max:
                       earth=True #符合条件才会继续寻找相应的拾取台站到时信息
                       gmttime = bjtime-8*3600
                       earth_num+=1
                       fa=open(out_name,'a+')
                       fa.write(line)
                       fa.close()
                   else :
                       earth=False
               if earth:
                   if part[2] in sta_list and (line[24:28].split()[0] in phase_list):
                       fa=open(out_name,'a+')
                       fa.write(line)
                       fa.close()
           except IndexError:
               continue
   
print ('total earthquake ==',earth_num)
print ('total phase      ==',phase_num)






















