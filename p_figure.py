#!/usr/bin/python
# -*- coding:UTF-8 -*-


import numpy as np
import matplotlib.pyplot as plt

def plot_num_distribution(num_list,fig_name,divnum,max_x,xlab,ylab): #画不同震级的数量分布图.输入不同的震级形成的列表,处数，震级最大值,横纵坐标轴名称
    x_list,y_list,z_list,new_x = [],[],[],[]  
    #最终的z_list会存放大于最大值的和。
    for k,g in groupby(sorted(num_list),key=lambda x:int(x//divnum)): #统计震级的数量，x_list整数震级，y是各个震级的数量。
        x_list.append(k)
        y_list.append(len(list(g)))
        
    
    for  i in range(len(x_list)):
        if x_list[i]>max_x:
            max_index = i #最大值所在索引
            break
    z = y_list[max_index]
    for j in range(max_index+1,len(x_list)):
        x_list.remove(x_list[max_index+1])
        z+= y_list[j]
    for i in range(max_index):
        z_list.append(y_list[i])
    z_list.append(z)
    
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


if __name__=="__main__":
    #1.散点图，读取txt文件中的第n列数据作为纵坐标，根据纵坐标的个数生成横坐标。
    #纵轴显示：找到y_list最大值(比如0.47)，据此生成间隔为0.1的y_show，[0,0.1,0.2,0.3,0.4]，然后把最大值添加进去[0.47]
    fa = open('zre.txt','r')
    a1 = fa.readlines()
    fa.close()
    x_list,y_list,y_show = [],[],[] #横坐标是根据y_list生成的一定范围的整数，y_show是展示的纵坐标。
    #读取txt文件中的第n列，存储纵坐标数据。
    for line in a1:
        co      = line.split() 
        y_value = float(co[0])
        y_list.append(y_value)
    #获得纵坐标最大值，用来显示在图中
    y_max = max(y_list)

    #根据纵坐标的个数生成n个整数    
    for i in range(len(y_list)):
        x_list.append(i)
        
    #修改纵坐标的显示(最大值是0.47,乘以10然后取整，得到0.4)
    for i  in np.arange(0,int(y_max*10)/10+0.1,0.1):
        y_show.append(round(i,2))
    y_show.append(y_max)
    
    #画图
    plt.figure(figsize=(25,15))
    ax=plt.subplot(1,1,1)#行数，列数，第几个图
    #显示y轴显示的数据
    ax.set_yticks(y_show)
    ax.set_yticklabels(y_show,rotation=0,fontsize=20)
    #画图主要命令
    plt.scatter(x_list,y_list,c='g',s=20,label='1')
    plt.savefig('test1')
    plt.show()    
    plt.close()
    #print (y_list)
    
    
    
    
    
    
    
    
    
    
    
    
    
