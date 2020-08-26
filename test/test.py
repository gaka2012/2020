#!/usr/bin/python
# -*- coding:UTF-8 -*-

import numpy as np
import glob,os
import matplotlib.pyplot as plt





#读取画图PhaseNet-master/dataset/waveform_train目录下的数据(npz格式的数据)
npz_datas = glob.glob('/home/zhangzhipeng/software/github/2020/test/*.npz')
for npz in npz_datas:
    file_name = os.path.basename(npz)
    file_name = file_name.replace('.npz','')
    print (npz,file_name)
    A = np.load(npz)
    ca_names = A.files
    print (ca_names)
    data = A[ca_names[0]]
    print (data.shape[1])
    print (data[:,0].shape)
  
    
'''
#画npz数据图，默认的是9001,3 的数据格式，保存的图片的名称是file_name
def plot_npz(file_name,data): 
    plt.figure(figsize=(25,15))
    ax=plt.subplot(data.shape[1],1,1) #(3,1,1) 输入的数据shape是9001,3
    for j in range(data.shape[1]):
        plt.subplot(data.shape[1],1,j+1,sharex=ax)
        t=np.linspace(0,data.shape[0]-1,data.shape[0]) #(0,9000,9001)
        plt.plot(t,data[:,j])
    plt.suptitle(png_name,fontsize=25,color='r')
    png_name=file_name+'.png' 
    plt.savefig(png_name)
    #os.system('mv *.png png') 
'''


'''
#1.1写入
#将数据村成npz格式的数据,默认的名字是arr_0,arr_1,arr_2等名字，也可以自定义为c_array
a = np.arange(3)
b = np.arange(4)
c = 12.0  #本来是整性，转换成npz后就变成了numpy
np.savez('test.npz',a,b,c_array=c)
print (a,b,c,type(c))

#1.2 读取
#A = np.load('test.npz')
#显示npz文件中有几个目录
catalog_names = A.files

print (catalog_names)
#print (type(A[catalog_names[2]]))
'''
