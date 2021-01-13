#!/usr/bin/python
# -*- coding:UTF-8 -*-

import os,glob,json,csv,re
from obspy.core import read
from datetime import *
from obspy.core import UTCDateTime
import numpy as np
import matplotlib.pyplot as plt
from itertools import groupby
from scipy import signal

#phasenet 5.1
#将cut_data中截取后的长度为120s的sac数据(3个分量)转换为npz数据，shape是3000，3，转存到npz_data中。并生成相应的csv文件
'''
data_path  = '/bashi_fs/centos_data/zzp/data' #截取过的长度为18001个点的sac三分量文件 
npz_path   = '/bashi_fs/centos_data/zzp/PhaseNet-master/dataset/ten_station_pred'   #将sac三分量转存成npz数据后保存置。

data_num = 0
data_files = sorted(glob.glob(data_path+'/*BHZ.sac'))
save_dict = {} #字典，存储文件名和P、S到时
os.chdir(data_path)
for data_file in data_files:
    z_channel = os.path.basename(data_file) #获得z分量的数据文件名称，前面已经修改过路径了！！SC.AXI_20180131230819.BHZ.sac
    #根据z分量的名称读取n、e分量,获得tp，ts到时及其对应的点数。
    st = read(z_channel.replace('BHZ.sac','*'))
    st.sort(keys=['channel'], reverse=False) #对三分量数据排序
        
    tp = st[0].stats.sac.a
    ts = st[0].stats.sac.t0
    b  = st[0].stats.sac.b  
    tp_num = int((tp-b)*100)-10500 #p波到时的点数是P波到时减去b值，参考时刻是发震时刻，p波到时的点数应该是1500，因为截取的时候是前15s
    ts_num = int((ts-b)*100)-10500 #s波到时有负数，比如-1234，再乘以100
    p_s_list = [tp_num,ts_num]

    try:
        #将数据转存成npz格式的3000,3的shape
        data = np.asarray(st.copy())[:,10500:13500].T

        #字典，存储每个npz文件的名称，以及对应的t、s到时点数。
        save_name = z_channel.replace('BHZ.sac','npz')
        save_dict.setdefault(save_name,[]).append(p_s_list)
        
        #保存成npz数据格式，名称是将后面的BHZ.sac替换成npz
        np.savez(save_name,data=data)
        os.system('mv *.npz %s'%(npz_path))
        data_num+=1
    except IndexError:
        print (data_file)
    
    
os.chdir('/bashi_fs/centos_data/zzp/PhaseNet-master/dataset')
f = open('ten_station.csv','w',encoding='utf-8')
csv_writer = csv.writer(f)
csv_writer.writerow (['fname','itp','its'])  #写入表头
for key,value in save_dict.items():
    #print (key,value[0][0])
    csv_writer.writerow([key,value[0][0],value[0][1]]) #写入实际数据
f.close()   

print('convert %s sac data to npz dat'%(data_num))
'''



#标签 phasenet 6.3.4对比自动拾取的结果和人工拾取的结果,针对地震事件波形数据，长度裁减为30s了


def plot_bar(right_list,left_list): #画柱状图，先把左右图的数据准备好。
    #right_list = [27, 7, 4, 2, 2, 1, 3, 0, 0, 1, 0, 0, 2, 1] #右半部分y轴数据
    a = [0+i*0.2 for i in range(14)]  #横坐标是0-13,大于13的不要了
    x = np.array(a)  #x轴坐标，间隔是0.2

    #left_list = [7, 1, 1, 1, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0] #左半部分y轴数据
    b = [0-i*0.2 for i in range(14)]
    x1= np.array(b)

    x_label = [-2,-1,0,1,2] #在哪个位置写x轴标签
    y_label = [100,200,300,400,500]

    width=0.2
    fig,ax = plt.subplots(figsize=(25, 15), dpi=100) #设置像素
    test1 = ax.bar(x+width/2,right_list,width,color='lightgreen',edgecolor='black') #画2遍，第一遍画右半部分的图像，默认是画在横坐标的中间，加上width/2后就画在了靠右边一点。
    test2 = ax.bar(x1-width/2,left_list,width,color='lightgreen',edgecolor='black')#第二遍画左边的图像。
   
    #在柱状图上添加数字
    for a,b in zip(x+width/2,right_list):
        plt.text(a, b+0.3,'%d'%b, ha = 'center',va = 'bottom',fontsize=15)   
    for a,b in zip(x1-width/2,left_list):
        plt.text(a, b+0.3,'%d'%b, ha = 'center',va = 'bottom',fontsize=15)
       
       
    #plt.bar(x,num_list,color='red',tick_label=name_list,width=0.1) #柱状图的宽度
    plt.xticks(x_label,('-1','-0.5','0','0.5','1'))  #写x轴标签
    plt.yticks(y_label)
    plt.tick_params(labelsize=15) #设置xy轴的字体大小。
    plt.xlabel('time residual (s)',fontsize=15)
    plt.ylabel('number of picks',fontsize=15)
    name = sum(right_list)+sum(left_list)
    plt.title(str(name),fontsize=24,color='r')
    
    plt.savefig('result.png')
    plt.show()
    plt.close()

def add_zero(a):#检查列表，正常应该是0-13每个数字对应一个number，对于没有number的数字设为0,大于13的予以剔除。   
    y = [] #画图时存储y值。
    count = 0
    i = 0
    try:
        for num in range(14):  #检查列表，正常应该是0-13每个数字对应一个number，对于没有number的数字设为0,大于13的予以剔除。   
            if count==14:
                break
            elif a[i][0]==count:
                y.append(a[i][1])
            elif a[i][0]!=count:
                y.append(0)
                i-=1
            count+=1
            i+=1
    except IndexError:
        for n in range(count,14):
            y.append(0)
    return y
    
#输入一个列表，里面是元组，取元组的第一个数，如果是正的，添加到pos列表中，如果是负的，添加到neg中，然后对2个列表进行排序以及分割，间距是0.1,需要调用add_zero函数。
#最后返回的是已经处理好的可以用于画图的列表，列表中的数据间隔是0.1(1)
def sta_list(tuple_list):
    pos,neg = [],[] #正数和负数分成2个列表
    right,left = [],[] #统计正数间隔为0.1的个数，以及负的间隔为0.1的个数
    
    for i in tuple_list:
        if i[0]>=0:
            pos.append(i[0])
        else:
            neg.append(i[0])
            
    for k,g in groupby(sorted(pos),key=lambda x:x*10//1): #统计正数中每个数字的个数。
        #print (k,len(list(g)))
        tup = (k,len(list(g))) #0.0代表范围是0-0.1,  1.0代表范围是0.1-0.2
        right.append(tup)
        
    for k,g in groupby(sorted(neg),key=lambda x:x*(-10)//1): #统计负数中每个数字的个数。
        #print (k,len(list(g)))
        tup = (k*(-1),len(list(g))) #0.0代表范围是0-0.1,  1.0代表范围是0.1-0.2
        left.append(tup)
    
    #最终得到的right1和left1是经过补零的从0-13的横坐标对应的纵坐标
    right1 = add_zero(right)
    left = sorted([(i[0]*-1,i[1]) for i in left],key=lambda x:x[0])   #将列表中的负数乘以负1,并重新排序
    left1 = add_zero(left)
    return right1,left1

#FP 7.3 read picks.csv and picks data which the time_residual less than 0.5 and generate a json file. 
out_name  = '/bashi_fs/centos_data/zzp/PhaseNet-master/output/picks.csv' #存储结果的csv文件
AI_right_list = []  #拾取的误差在0.5s的文件名称

out = [] #存储所有的结果，用来画图
with open(out_name) as f:
    reader = csv.reader(f)  #读取csv文件内容
    header = next(reader)   #返回文件的下一行，类型是列表
   
    #遍历所有AI拾取，如果没有拾取到就复制到1000_data中的no_pick_data，如果拾取的误差大于1.4s，就复制到wrong_pick_data
    for row in reader:
        if row[1]=='[]':
            pass 
        else:
            pick_nums = re.findall('\d+',row[1]) #row[1]是str--'[1448]'，找到其中的数字，并转换成整数，有的有2个拾取
            if abs(int(pick_nums[0])-1499) < 140:
                sub = int(pick_nums[0])-1499
                sub_tuple = (sub/100,1)
                AI_right_list.append(row[0])
                out.append(sub_tuple)

filename = 'AI_right_list.json'
with open(filename,'w') as file_obj:
    json.dump(AI_right_list,file_obj)    
    
print (len(AI_right_list))         


right,left = sta_list(out)  #处理统计时间差，将其整理好，以备画图时用，只保留误差在0.13-0.14(1.3)之下的，其他的不要了。
plot_bar(right,left)           #画柱状图






'''
#phasenet 7.5 重新生成新的csv文件，不用重新生成npz，因为AI是根据csv读取数据的。
data_path  = '/bashi_fs/centos_data/zzp/data' #截取过的长度为18001个点的sac三分量文件 
npz_path   = '/bashi_fs/centos_data/zzp/PhaseNet-master/dataset/ten_station_pred'   #将sac三分量转存成npz数据后保存置。

data_num = 0
data_files = sorted(glob.glob(data_path+'/*BHZ.sac'))
save_dict = {} #字典，存储文件名和P、S到时
os.chdir(data_path)
for data_file in data_files:
    z_channel = os.path.basename(data_file) #获得z分量的数据文件名称，前面已经修改过路径了！！SC.AXI_20180131230819.BHZ.sac
    #根据z分量的名称读取n、e分量,获得tp，ts到时及其对应的点数。
    st = read(z_channel.replace('BHZ.sac','*'))
    st.sort(keys=['channel'], reverse=False) #对三分量数据排序
        
    tp = st[0].stats.sac.a
    ts = st[0].stats.sac.t0
    b  = st[0].stats.sac.b  
    tp_num = int((tp-b)*100)-10500 #p波到时的点数是P波到时减去b值，参考时刻是发震时刻，p波到时的点数应该是1500，因为截取的时候是前15s
    ts_num = int((ts-b)*100)-10500 #s波到时有负数，比如-1234，再乘以100
    p_s_list = [tp_num,ts_num]

    try:

        #字典，存储每个npz文件的名称，以及对应的t、s到时点数。
        save_name = z_channel.replace('BHZ.sac','npz')
        save_dict.setdefault(save_name,[]).append(p_s_list)
        data_num+=1
    except IndexError:
        print (data_file)
    
    
os.chdir('/bashi_fs/centos_data/zzp/PhaseNet-master/dataset')
f = open('ten_station.csv','w',encoding='utf-8')
csv_writer = csv.writer(f)
csv_writer.writerow (['fname','itp','its'])  #写入表头
for key,value in save_dict.items():
    #print (key,value[0][0])
    csv_writer.writerow([key,value[0][0],value[0][1]]) #写入实际数据
f.close()   

print('convert %s sac data to npz dat'%(data_num))

'''























