#!/usr/bin/python
# -*- coding:UTF-8 -*-


import glob,re,os
from obspy.core import read
from datetime import *
from obspy.core import UTCDateTime





'''
a = ['xxx.BHE.sac','xxx.BHZ.sac','xxx.BHZ.sac']

gdata = '/home/zhangzhipeng/datatest/SC/AXI/2018/01/04'

test = os.listdir(gdata)
test1 = '_'.join(a)
#os.path.exists(gdata)
print(test,test1)
if 'BHE' not in test1 or 'BHN' not in test1 or  'BHZ' not in test1:
    print('ttest')
'''


'''
st = read('/home/zhangzhipeng/software/github/2020/program/etest/*.sac')
st.sort(['starttime'])
for tr in st:
    s = tr.stats.sac
    r_time = UTCDateTime(year=s.nzyear,julday=s.nzjday,hour=s.nzhour,minute=s.nzmin,second=s.nzsec,microsecond=s.nzmsec*1000) #参考时刻
    print (tr.stats.starttime,r_time,s.b,s.a,s.evla,s.dist,s.mag,s.az)
'''

'''
st = read('SC*')
test = st[0]
print (test.stats.sac)
c_st = test.copy()

dt   = UTCDateTime("2017-12-31T23:58:12")
ot = UTCDateTime("2017-12-31T23:55:10.70") #发震时刻

data = c_st.trim(dt, dt + 30,pad=True, fill_value=0)   #默认pad是false,这里是True，当给定的截取时间超出了数据范围会自动补
s = data.stats.sac

r_time = UTCDateTime(year=s.nzyear,julday=s.nzjday,hour=s.nzhour,minute=s.nzmin,second=s.nzsec,microsecond=520*1000) #参考时刻
#print (ot.microsecond/1000)
s.nzyear,s.nzjday,s.nzhour,s.nzmin,s.nzsec,s.nzmsec = ot.year,ot.julday,ot.hour,ot.minute,ot.second,int(ot.microsecond/1000) #发震时刻设为参考时刻
#,s.nzjday,s.nzhour,s.nzmin,s.nzsec,s.nzmsec = ot.year,ot.julday,ot.hour,ot.minute,ot.second,ot.microsecond

print (s.nzmsec)
#print (s.nzyear,s.nzjday,s.nzhour,s.nzmin,s.nzsec,s.nzmsec)

#s.evla,s.evlo = 12,31
s.az = 31.2

data.write('test.sac',format='SAC')



new_st = read('test.sac')
print (new_st[0].stats.starttime)
print (new_st[0].stats.sac)
#print (new_st[0].stats.sac.a)
#print (dir(new_st[0].stats.sac.keys))
#int(ot.microsecond/1000)

'''

'''
data[0].stats.network = 'BJ'
data[0].stats.sac.b   = 10
data[0].stats.sac.nzyear   = 2018
data.write('test2.sac',format='SAC')

print (data[0].stats.sac.b)

print (st)


st = read('test2.sac')
print (st[0].stats.sac.b)
'''




'''
st = stream.Stream()
print (dir(st))  #stream.Stream实际上一个在文件stream.py中的一个类，dir会显示这个类的属性，以及其函数。
print (help(st.merge)) #help会显示类st中的方法(函数)merge的用法。
'''

'''
import matplotlib.pyplot as plt
from itertools import groupby

def plot_num_distribution(num_list,fig_name): #画不同震级的数量分布图.
    x_list,y_list = [],[]  #输入不同的震级形成的列表
    for k,g in groupby(sorted(num_list),key=lambda x:int(x//1)): #统计震级的数量，x_list是拥有的整数震级，y是各个震级的数量。
        x_list.append(k)
        y_list.append(len(list(g)))
    
    plt.figure(figsize=(25, 15))
    #print (dir(plt))
    plt.bar(x_list,y_list,color='blue',width=0.1) #柱状图的宽度
    plt.title(fig_name,fontsize=24,color='r')
    plt.tick_params(labelsize=23)
    plt.savefig(fig_name)  #注意要在plt.show之前
    #plt.show()
    plt.close()

num_list = [0.1,0.3,1,2,3,4,5,1.2,1,1] #y轴数据         
name = 'test'
plot_num_distribution(num_list,name)

'''











