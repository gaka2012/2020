#!/usr/bin/python
# -*- coding:UTF-8 -*-

import glob
from obspy.core import read
from datetime import *
from obspy.core import UTCDateTime

f1=read('head.SAC')
tr=f1[0]
print (tr.data[141700])
#fa=open('00.txt')

