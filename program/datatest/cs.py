#!/usr/bin/python
# -*- coding:UTF-8 -*-


import glob,re,os
from obspy.core import read
from datetime import *
from obspy.core import UTCDateTime


'''
st = read('/home/zhangzhipeng/software/github/2020/program/datatest/*.sac')
st.sort(['starttime'])
for tr in st:
    s = tr.stats.sac
    r_time = UTCDateTime(year=s.nzyear,julday=s.nzjday,hour=s.nzhour,minute=s.nzmin,second=s.nzsec,microsecond=s.nzmsec*1000) #参考时刻
    p_time = r_time+s.a
    print (p_time)
'''

#st = read('/home/zhangzhipeng/software/github/2020/program/sac_data/SC.AXI_20180120121318.BHZ.sac')
#print (st[0].stats.sac.t0)

st = read('*.sac')
print (st[0].stats.sac)

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















