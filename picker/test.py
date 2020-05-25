#!/usr/bin/python
# -*- coding:UTF-8 -*-

from obspy.core import read
from obspy import UTCDateTime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdate
from datetime import *  

st = read('SC.AXI_20180101062906.BHZ.sac')

s= st[0].stats.sac
dt1 = UTCDateTime(year=s.nzyear,julday=s.nzjday,hour=s.nzhour,minute=s.nzmin,second=s.nzsec,microsecond=s.nzmsec)

print (dt1+s.a)





new_list = []
time = ['20140901-06370010','20140902-06370020','20140906-06370030']
for i in time:
    st_time = datetime.strptime(i,"%Y%m%d-%H%M%S%(2f)")
    print (help(st_time.strptime))
    new_list.append(st_time)
for i in new_list:
    print (i,type(i))
y    = [1,2,3]

'''
plt.figure(figsize=(10, 8))
ax = plt.subplot(111)
ax.set_xticks(new_list)
ax.set_xticklabels(new_list,rotation=70,fontsize=10)
ax.xaxis.set_major_formatter(mdate.DateFormatter('%Y/%m/%d %H:%M:%S.%f'))
#ax.xaxis.set_major_formatter(mdate.DateFormatter('yyyy-MM-dd HH:mm:ss'))
#print (help(ax.xaxis.set_major_formatter))
ax.plot(new_list,y,linewidth='2',label='a',color='blue') #设置线条的宽度，标签，颜色。
plt.title('test')
plt.legend()
plt.show()
plt.close()
'''
