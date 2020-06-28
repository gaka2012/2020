#!/usr/bin/python
# -*- coding:UTF-8 -*-

import glob,os
from obspy.core import UTCDateTime
import matplotlib.pyplot as plt
from itertools import groupby


#[3.0, 3.0, 5.0, 5.0, 8.0, 11.0, 14.0, 17.0, 21.0, 22.0]
#[0, 1, 2] [5, 3, 2] 
#divnum,max_x = 10,10 大于等于最大值的放在一起
#[0, 1] [5, 5]

def plot_num_distribution(num_list,fig_name,divnum,max_x,xlab,ylab): #画不同震级的数量分布图.输入不同的震级形成的列表,处数，震级最大值,横纵坐标轴名称
    x_list,y_list,z_list,new_x = [],[],[],[]  
    #最终的z_list会存放大于最大值的和。
    for k,g in groupby(sorted(num_list),key=lambda x:int(x//divnum)): #统计震级的数量，x_list整数震级，y是各个震级的数量。
        x_list.append(k)
        y_list.append(len(list(g)))
        
    print (x_list,y_list,max_x/divnum)
    
    for  i in range(len(x_list)):
        if x_list[i]>=(max_x/divnum):
            max_index = i #最大值所在索引
            break
    z = y_list[max_index]
    for j in range(max_index+1,len(x_list)):
        x_list.remove(x_list[max_index+1])
        z+= y_list[j]
    for i in range(max_index):
        z_list.append(y_list[i])
    z_list.append(z)
    
    print (x_list,z_list)
    
    new_x = [divnum*i for i in x_list]
    #画图
    plt.figure(figsize=(25, 15))
    plt.bar(new_x,z_list,color='blue',width=0.6) #柱状图的宽度
    plt.title(fig_name,fontsize=24,color='r')
    plt.tick_params(labelsize=23)
    plt.xticks(new_x)
    plt.xlabel(xlab,fontsize=30) #加上横纵坐标轴的描述。
    plt.ylabel(ylab,fontsize=30) #
    plt.savefig(fig_name)  #注意要在plt.show之前
    #plt.show()
    plt.close()



#遍历所有数据名称，获得npz_list列表，里面是文件名去掉npz
npz_list = glob.glob('/home/zhangzhipeng/software/github/2020/figure/*.npz')
npz_list = [(os.path.basename(i)).replace('.npz','') for i in npz_list]  #SC.AXI_20180127112142_EV


#打开文件进行遍历，如果与上面的列表吻合，则将数据挑出来画图。 
fa  = open('2018.dat','r')
con = fa.readlines()
fa.close()

dep_list = []
mag_list = []
dis_list = []
dif_list = []

ear_num  = 0
for line in con:
    net,sta,p,pt,s,st,o,ot,lon,lat,dep,mag,dist = line.split(',')
    #根据文档中的内容生成文件名，看一下是否在图像文件列表中
    year,mon,end = ot.split('-')
    day,end = end.split('T')
    hour,minu,sec = end.split(':')
    sec = sec.split('.')[0]
    ot = "{}{}{}{}{}{}".format(year,mon,day,hour,minu,sec) #
    new_name = "{}.{}_{}_EV".format(net,sta,ot)
    
    #如果文件名在列表中，将数据挑出来画图。
    #if new_name in npz_list:
    if 2>1:
        try:
            pt  = UTCDateTime(pt)
            st  = UTCDateTime(st)
            t_d = st-pt
            dif_list.append(t_d)
            
            dep_list.append(float(dep))
            mag_list.append(float(mag))
            dis_list.append(float(dist))

        except ValueError:
            pass
dep_list = sorted(dep_list)
mag_list = sorted(mag_list)
dis_list = sorted(dis_list)
dif_list = sorted(dif_list)
#print (dep_list)

print (dis_list[-10:])
print (dif_list[-10:])

#print (dep_list[-10:-1])    

plot_num_distribution(dep_list,'dep_dis',10,10,'dep','count') 

















