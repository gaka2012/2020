#!/usr/bin/python
# -*- coding:UTF-8 -*-


import matplotlib.pyplot as plt
import numpy as np

#画npz数据图，默认的是9001,3 的数据格式，保存的图片的名称是file_name,输入的数据是文件名，数据，p和s到时(用来画竖线)
def plot_npz(file_name,data,itp,its): 
    plt.figure(figsize=(25,15))
    ax=plt.subplot(data.shape[2],1,1) #(3,1,1) 输入的数据shape是9001,3
    for j in range(data.shape[2]):
        plt.subplot(data.shape[2],1,j+1,sharex=ax)  #(3,1,1) (3,2,1) (3,3,1)
        t=np.linspace(0,data.shape[0]-1,data.shape[0]) #(0,9000,9001)
        #plt.vlines(itp,min(data[:,j]),max(data[:,j]),colors='r') 
        #plt.vlines(its,min(data[:,j]),max(data[:,j]),colors='r')  
        plt.plot(t,data[j,:,:].flatten())
    
    plt.suptitle(file_name,fontsize=25,color='r')
    png_name=file_name+'.png' 
    plt.savefig(png_name)
    #os.system('mv *.png png') 
    plt.close()


a = np.array([1,2,3,4,5,6,7,8,9])

b = np.reshape(a,[3,3]) #重新定义成2行3列。
c = b[:,np.newaxis,:]  #增加一个维度，shape变成了(5,1)
#print (c,c.shape)

d = c[0,:,:]



plot_npz('plot_test',c,12,12)
