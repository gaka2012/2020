#!/usr/bin/python
# -*- coding:UTF-8 -*-

import os,glob,json,csv
from obspy.core import read
from datetime import *
from obspy.core import UTCDateTime
import numpy as np
import matplotlib.pyplot as plt


#script1.1 
#将cut_data中截取后的长度为30s的sac数据(3个分量)转换为npz数据，shape是3000，3，转存到npz_data中。并生成相应的csv文件
data_path  = '/home/zhangzhipeng/data/cuted_data' #截取过的长度为3001个点的sac三分量文件 
npz_path   = '/home/zhangzhipeng/data/npz_data'   #将sac三分量转存成npz数据后保存位置。

data_files = sorted(glob.glob(data_path+'/*BHZ.sac'))
save_dict = {} #字典，存储文件名和P、S到时
os.chdir(data_path)
for data_file in data_files:
    z_channel = os.path.basename(data_file) #获得z分量的数据文件名称，前面已经修改过路径了！！SC.AXI_20180131230819.BHZ.sac
    #根据z分量的名称读取n、e分量,获得tp，ts到时及其对应的点数。
    st = read(z_channel)
    st+= read(z_channel.replace('BHZ','BHN'))
    st+= read(z_channel.replace('BHZ','BHE'))
    tp = st[0].stats.sac.a
    ts = st[0].stats.sac.t0
    b  = st[0].stats.sac.b  
    tp_num = int((tp-b)*100) #p波到时的点数是P波到时减去b值，参考时刻是发震时刻，p波到时的点数应该是1500，因为截取的时候是前15s
    ts_num = int((ts-b)*100) #s波到时有负数，比如-1234，再乘以100
    p_s_list = [tp_num,ts_num]

    #字典，存储每个npz文件的名称，以及对应的t、s到时点数。
    save_name = z_channel.replace('BHZ.sac','npz')
    save_dict.setdefault(save_name,[]).append(p_s_list)

    #将sac数据转化为numpy格式的数据，由于切割的时候是3001个点，只取前3000个点。
    e_cha = np.asarray(st[2].copy())[:3000]
    n_cha = np.asarray(st[1].copy())[:3000]
    z_cha = np.asarray(st[0].copy())[:3000]
    
    #将3个分量的数据合并，shape转换为3000,3
    data = np.vstack([e_cha,n_cha,z_cha]).T
        
    #保存成npz数据格式，名称是将后面的BHZ.sac替换成npz
    save_name = z_channel.replace('BHZ.sac','npz')
    np.savez(save_name,data=data)
    os.system('mv *.npz %s'%(npz_path))
    #print (data.shape)
    
os.chdir('/home/zhangzhipeng/temp_delete')
f = open('test.csv','w',encoding='utf-8')
csv_writer = csv.writer(f)
csv_writer.writerow (['fname','itp','its'])  #写入表头
for key,value in save_dict.items():
    print (key,value[0][0])
    csv_writer.writerow([key,value[0][0],value[0][1]]) #写入实际数据
f.close()   
  

    
'''
os.chdir('/home/zhangzhipeng/temp_delete') 
filename='dict.json'
with open(filename,'w') as file_obj:
    json.dump(save_dict,file_obj)


filename='dict.json'
with open(filename) as file_obj:
    p_s_dict = json.load(file_obj)
for key,value in p_s_dict.items():
    print (key,value[0])
''' 
    

#1.2npz三分量格式数据画图，shape=3000*3,无P、S到时
'''
def plot_waveform_pred(plot_dir,file_name,data): 
    plt.figure(figsize=(25,15))
    ax=plt.subplot(data.shape[1],1,1) #(3,1,1) 输入的数据shape是3000,3
    for j in range(data.shape[1]):
        plt.subplot(data.shape[1],1,j+1,sharex=ax)
        t=np.linspace(0,data.shape[0]-1,data.shape[0]) #(0,2999,3000)
        plt.plot(t,data[:,j])
    
    plt.suptitle(file_name,fontsize=25,color='r')
    png_name=file_name+'.png' 
    plt.savefig(plot_dir+png_name)
    #os.system('mv *.png png') 
    plt.close()    
    
#读取画图PhaseNet-master/dataset/waveform_pred目录下的数据(npz格式的数据)
npz_datas = glob.glob('/home/zhangzhipeng/data/npz_data/*.npz')
for npz in npz_datas:
    #获得npz数据的文件名，将后缀.npz去掉，作为参数传递给画图子函数。
    file_name = os.path.basename(npz)
    file_name = file_name.replace('.npz','')
    
    #获得npz数据中的data,是一个三分量的数据，作为参数传递给画图子函数。   
    A = np.load(npz)
    ca_names = A.files
    data = A[ca_names[0]]    
    plot_waveform_pred('/home/zhangzhipeng/data/npz_figure/',file_name,data)
'''






























