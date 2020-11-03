#!/usr/bin/python
# -*- coding:UTF-8 -*-


import os,sys,glob
from obspy.core import UTCDateTime
import csf_phase
from datetime import *   
import matplotlib.dates as mdate
import numpy as np
import matplotlib.pyplot as plt



#1.2 读取
name_path = '/home/zhangzhipeng/data/npz_data/*.npz' #存放npz数据的目录
name_file = glob.glob(name_path)

#对npz文件画图，输入保存路径、3分量数据、tp、ts
def plot_npz(fig_out_path,data,tp,ts):
    plt.figure(figsize=(25,15)) 
    fig, ax = plt.subplots(3,1,sharey='all',sharex=True)
    for t in range(data.shape[0]):
        ax[t].plot(data[t],color='red')
        ax[t].set_xlabel('time (samples)')
        ax[t].axvline(x=tp,color='green') #画P波到时
        if ts=='null':
            pass
        else:
            ax[t].axvline(x=ts,color='blue')  #
    ax[0].set_title('raw waveform')
    plt.savefig(fig_out_path)
    plt.close() 

for name in name_file:
    read_data  = np.load(name)
    data,tp,ts = read_data['data'],read_data['itp'],read_data['its'] 
    fig_path   = name[:-4]+'.png'
    fig_path   = fig_path.replace('2018test','2018png')
    plot_npz(fig_path,data,tp,ts)
    '''
    
    
    
    
