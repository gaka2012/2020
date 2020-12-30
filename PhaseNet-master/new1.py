#!/usr/bin/env python
#-*- coding:utf-8 -*-


import glob,os
from obspy.core import read


#data_path = '/bashi_fs/centos_data/zzp/data/*.BHZ.sac'


#统计地震台站信息

'''
data_path = '/home/zhangzhipeng/software/sac_data/*.BHZ.sac'
data_files = glob.glob(data_path)

sta_list = []
sta_dict = {}
for data_file in data_files:
    name = os.path.basename(data_file)
    st_sta = name.split('_')[0]
    if st_sta not in sta_list:
        st = read(data_file)
        stats = st[0].stats.sac
        info_list = [stats.stla,stats.stlo]
        sta_dict.setdefault(st_sta,[]).extend(info_list)
                
        sta_list.append(st_sta)

fa = open('sta_lat.txt','a+')
for key,value in sta_dict.items():
    fa.write(key+' '+str(value[0])+' '+str(value[1])+'\n')
fa.close()
'''

data_path = '/home/zhangzhipeng/software/sac_data/*.BHZ.sac'
data_files = glob.glob(data_path)

sta_list = []
sta_dict = {}
for data_file in data_files[:1]:
    name = os.path.basename(data_file)
    st_sta = name.split('_')[0]
    if st_sta not in sta_list:
        st = read(data_file)
        stats = st[0].stats.sac
        print(stats)                

















    
