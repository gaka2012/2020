#!/usr/bin/python
# -*- coding:UTF-8 -*-


from obspy.core import read
from datetime import *
from obspy.core import UTCDateTime
import glob

mfiles = glob.glob('/home/zhangzhipeng/software/github/2020/test/mseed/*.mseed')
st =  read(mfiles[0])           # BHE
#st += read(mfiles[1]) 

print (st) 

st.plot()
