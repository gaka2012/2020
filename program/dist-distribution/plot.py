#!/usr/bin/python
# -*- coding:UTF-8 -*-


from obspy.core import read
import glob,os
from obspy.core import stream

data_path = '/home/zhangzhipeng/software/github/2020/program/dist-distribution/sac_data_o/SC'
sac_files = glob.glob(data_path+'/*/2017*/*/*/*.BHZ.SAC')

for sac_file in sac_files:
    st = stream.Stream()
    st+= read(sac_file)
    h0 = st[0].stats.starttime
    h1 = st[0].stats.endtime
    name = (os.path.basename(sac_file)).replace('SAC','jpg')
    st.plot(starttime= h0 ,endtime=h1,equal_scale=False,outfile=name,color='red',size=(1800,1000)) #单张图，这个size比较好，三张图一起画，默认size就行。
