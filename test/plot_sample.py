#!/usr/bin/python
# -*- coding:UTF-8 -*-

#启用obspy环境画图，否则容易报错。

import glob,os
import numpy as np
import matplotlib.pyplot as plt


'''
#画截取后的数据图，默认的是的数据格式(3000,1,3)，保存的图片的名称是file_name,输入的数据是文件名，数据，p和s到时(用来画竖线)
def plot_npz(plot_dir,file_name,data,itp,its): 
    plt.figure(figsize=(25,15))
    ax=plt.subplot(data.shape[2],1,1) #(3,1,1) 输入的数据shape是(3000,1,3)
    for j in range(data.shape[2]):
        plt.subplot(data.shape[2],1,j+1,sharex=ax)  #(3,1,1) (3,2,1) (3,3,1)
        t=np.linspace(0,data.shape[0]-1,data.shape[0]) #(0,2999,3000)
        #plt.vlines(itp,min(data[:,j]),max(data[:,j]),colors='r') 
        #plt.vlines(its,min(data[:,j]),max(data[:,j]),colors='r')  
        plt.plot(t,data[:,0,j].flatten())
    
    plt.suptitle(plot_dir.split('/')[-2]+file_name,fontsize=25,color='r')
    png_name=file_name+'.png' 
    plt.savefig(plot_dir+png_name)
    #os.system('mv *.png png') 
    plt.close()
    


npz_dir_list = ['./sample/cut/','./sample/normali/','./sample/adjust_miss/']    
#读取画图目录sample/cut/下的数据(npz格式的数据),是经过截取的数据，数据格式由原来的(9001,3)变成了(3000,1,3)
for i in range(3):
    npz_dir = npz_dir_list[i]
    print (npz_dir)
    npz_datas = glob.glob(npz_dir+'*.npz')
    for npz in npz_datas:
        #获得npz数据的文件名，将后缀.npz去掉，作为参数传递给画图子函数。
        file_name = os.path.basename(npz)
        file_name = file_name.replace('.npz','')
        
        #获得npz数据中的data,是一个三分量的数据，作为参数传递给画图子函数。   
        A = np.load(npz)
        ca_names = A.files
        data = A[ca_names[0]]
        
        #画图
        plot_npz(npz_dir,file_name,data,12,12)   
    
''' 
    
    
    
    
    
    
    
    

'''
def normalize(data):
    data -= np.mean(data, axis=0, keepdims=True)
    std_data = np.std(data, axis=0, keepdims=True)
    assert(std_data.shape[-1] == data.shape[-1])
    std_data[std_data == 0] = 1
    data /= std_data
    return data
  

A = np.load('NC_BJOB_2017111323254117.npz')
n = A.files
data = A[n[0]] #(3000,1,3)
print (data.shape)

data = normalize(data)
  
plot_npz('./','nor',data,123,123)
'''  


#画原始npz数据图，默认的是9001,3 的数据格式，保存的图片的名称是file_name,输入的数据是文件名，数据，p和s到时(用来画竖线)
def plot_npz(plot_dir,file_name,data,itp,its): 
    plt.figure(figsize=(25,15))
    ax=plt.subplot(data.shape[1],1,1) #(3,1,1) 输入的数据shape是9001,3
    for j in range(data.shape[1]):
        plt.subplot(data.shape[1],1,j+1,sharex=ax)
        t=np.linspace(0,data.shape[0]-1,data.shape[0]) #(0,9000,9001)
        plt.vlines(itp,min(data[:,j]),max(data[:,j]),colors='r') 
        plt.vlines(its,min(data[:,j]),max(data[:,j]),colors='r')  
        plt.plot(t,data[:,j])
    
    plt.suptitle(file_name,fontsize=25,color='r')
    png_name=file_name+'.png' 
    plt.savefig(plot_dir+png_name)
    #os.system('mv *.png png') 
    plt.close()    
    

'''
#读取画图PhaseNet-master/dataset/waveform_train目录下的数据(npz格式的数据)
npz_datas = glob.glob('./sample/original/*.npz')
for npz in npz_datas:

    #获得npz数据的文件名，将后缀.npz去掉，作为参数传递给画图子函数。
    file_name = os.path.basename(npz)
    file_name = file_name.replace('.npz','')
    
    #获得npz数据中的data,是一个三分量的数据，作为参数传递给画图子函数。   
    A = np.load(npz)
    ca_names = A.files
    data = A[ca_names[0]]
    #获得itp和its的位置，用来在图上画竖线
    itp = A[ca_names[1]]  #是一个数字，比如3001
    its = A[ca_names[2]]
    
    plot_npz('./sample/original/',file_name,data,itp,its)
    
'''  
    
#标签2.2
#画给的waveform_pred中的数据，默认shape是3000,3  需要给定参数：保存路径、文件名称、数据
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
npz_datas = glob.glob('/home/zhangzhipeng/software/github/2020/PhaseNet-master/dataset/waveform_pred/*.npz')
for npz in npz_datas:
    #获得npz数据的文件名，将后缀.npz去掉，作为参数传递给画图子函数。
    file_name = os.path.basename(npz)
    file_name = file_name.replace('.npz','')
    
    #获得npz数据中的data,是一个三分量的数据，作为参数传递给画图子函数。   
    A = np.load(npz)
    ca_names = A.files
    data = A[ca_names[0]]    
    plot_waveform_pred('/home/zhangzhipeng/software/github/2020/test/waveform_pred/',file_name,data)
    
    
    
    
    
    
    
    
    
    
