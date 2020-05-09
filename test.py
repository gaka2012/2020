#!/usr/bin/python
# -*- coding:UTF-8 -*-

from datetime import *
import pandas as pd
from obspy.core import UTCDateTime
import os

start_time,end_time = '2015-07-01T00:00:00','2015-07-02T00:00:00'
begin_time = UTCDateTime(start_time)
#end_time   = UTCDateTime(end_time)+3
close_time = datetime.strptime(end_time,"%Y-%m-%dT%H:%M:%S")+timedelta(days=31)
print (type(close_time))
'''
df = pd.date_range(start = start_time.split('T')[0], end = end_time.split('T')[0], freq='M',normalize=True) #按月迭代，注意start和end中必须是年月日，迭代后的是每个月的月底日期，但是缺少最后一个月的。
df2= df.strftime('%Y-%m') #整理成格式 2017-08(str)
last = datetime.strftime(datetime.strptime(end_time,"%Y-%m-%dT%H:%M:%S")+timedelta(days=31),'%Y-%m')   
file_list = [i+'.txt' for i in df2]
file_list.append(last+'.txt') #['2015-07.txt', '2015-08.txt']

report_path = 'report'
for report_file in file_list:
    report_file = os.path.join(report_path,report_file)
    print (report_file)
    
'''
