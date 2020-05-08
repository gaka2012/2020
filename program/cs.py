#!/usr/bin/python
# -*- coding:UTF-8 -*-

import glob,re,os
from obspy.core import read
from datetime import *
from obspy.core import UTCDateTime
from obspy.core import stream
import numpy as np
import matplotlib.pyplot as plt
from datetime import *
import matplotlib.dates as mdate



'''
st = read('/home/zhangzhipeng/software/github/2020/program/sac_data/*.sac')
st.sort(['starttime'])
for tr in st:
    s = tr.stats.sac
    r_time = UTCDateTime(year=s.nzyear,julday=s.nzjday,hour=s.nzhour,minute=s.nzmin,second=s.nzsec,microsecond=s.nzmsec*1000) #参考时刻
    p_time = r_time+s.a
    print (tr.stats.starttime,r_time,p_time,s.a,s.evla,s.dist,s.mag,s.az) #输出绝对时间、参考时间、p到时

'''
