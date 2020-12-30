#!/usr/bin/env python
#-*- coding:utf-8 -*-

from obspy.core import UTCDateTime
import csv
import numpy as np
import time,os
import subprocess
from obspy.core import read
import glob,re
import matplotlib.pyplot as plt
from itertools import groupby


#sac文件中的详细信息，需要传入trace
class Sac_info():
    def __init__(self,tr):
        self.info = tr.stats.sac
        self.dist = self.info.dist
        self.mag  = self.info.mag
        
class Noise_process():
    def __init__(self,tr,noise_path,sac_name):
        if not os.path.exists(noise_path):
            os.makedirs(noise_path)
        data_starttime = tr.stats.starttime
        c_tr = tr.copy()
        data = c_tr.trim(data_starttime, data_starttime+90,pad=True, fill_value=0)     
        noise_name = noise_path+sac_name
        data.write(noise_name,format='SAC')


class Sac_files():
    def __init__(self,sac_file_path,get_noise=False):
        self.sac_file_path = sac_file_path
        self.sac_info_all = {}  #存储sac头文件的所有信息  
        self.get_noise = get_noise #是否截取噪声


        self.build() #实例化类直接运行这个函数,最好放在最后，让前面的属性先赋值完毕。
        
    def read_sac_BHZ(self):  #读取路径下的所有Z分量的SAC数据,实例化类的时候直接运行这个,获得self.sac_info_all.setdefault字典，存储头文件信息
        sac_files = glob.glob(self.sac_file_path+'/*.BHZ.sac')
        for sac_file in sac_files:
            sac_name = os.path.basename(sac_file)
            st = read(sac_file)
            self.sac_info = Sac_info(st[0]) #读取sac头文件中的所有信息
            tem_list = [self.sac_info.dist,self.sac_info.mag] #临时列表，存储头文件的震中距，震级
            self.sac_info_all.setdefault(sac_name,[]).extend(tem_list)         
            
            if self.get_noise: #剪切数据，长度为60s
                process_noise = Noise_process(st[0],self.sac_file_path+'/noise_data/',sac_name)
                
            
    def build(self):
        self.read_sac_BHZ()   #直接调用这个子函数。
        print ('bulid is running')
    
    def plot_sac(self,file_path,save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        sac_files = glob.glob(file_path+'/*.BHZ.sac')
        for sac_data in sac_files:
            name = os.path.basename(sac_data)
            st = read(sac_data) 
            save_name = save_path+name.replace('BHZ.sac','png')  
            st.plot(equal_scale=False,outfile=save_name,color='red',size=(1800,1000)) 
          
    

sac_data = Sac_files('/home/zhangzhipeng/software/github/2020/data',get_noise=True)




















