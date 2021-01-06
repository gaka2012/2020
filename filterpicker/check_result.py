#!/usr/bin/python
# -*- coding:UTF-8 -*-

import os,glob,json
import subprocess
from obspy.core import UTCDateTime
import matplotlib.pyplot as plt
from itertools import groupby
import numpy as np
from obspy.core import read



def plot_bar(right_list,left_list): #画柱状图，先把左右图的数据准备好。
    #right_list = [27, 7, 4, 2, 2, 1, 3, 0, 0, 1, 0, 0, 2, 1] #右半部分y轴数据
    a = [0+i*0.2 for i in range(14)]  #横坐标是0-13,大于13的不要了
    x = np.array(a)  #x轴坐标，间隔是0.2

    #left_list = [7, 1, 1, 1, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0] #左半部分y轴数据
    b = [0-i*0.2 for i in range(14)]
    x1= np.array(b)

    x_label = [-2,-1,0,1,2] #在哪个位置写x轴标签
    y_label = [100,200,300]

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
    
    
#画散点图，输入一个列表，列表中的元素是一个元组，里面是y轴值以及需要在y轴上显示的文字，输入一个最大值，限制了y轴最大值不能超过这个，否则会被设为最大值。
def plot_scatter(yz_list,max_y):
    y,z = [],[]  #纵坐标的数值与在纵坐标中显示的数字
    for ele in yz_list:
        if ele[0]>max_y:
            y.append(max_y)
            z.append(ele[1])
        else:
            y.append(ele[0])
            z.append(ele[1])
    x = [i for i in range(len(y))] #根据纵坐标生成横坐标
    print (y)

    plt.figure(figsize=(25,15))
    #画图
    plt.scatter(x,y,c='g',s=20,label='1') #c是颜色，s是画的点的大小

    #在折线图上加上数字
#    i=0
#    for a1,b1 in zip(x,y): #前面2个参数是位置，
#        plt.text(a1,b1+0.2,z[i],ha='center',va='bottom',fontsize=12)   #plt.text(1,1,c,ha='center',va='bottom',fontsize=7)
#        i+=1

    #横纵轴坐标的描述
    plt.xlabel('x',fontsize=25)
    plt.ylabel('y',fontsize=25)

    plt.savefig('test1')
    #plt.show()    
    plt.close()


#读取filterpicker生成的结果文件中的时间(zday1.txt)，将其中的6,7,8列转换为UTC时间格式。
#输入filterpicker生成的结果文件名称，输入标准答案，计算人工拾取与自动拾取的时间差，找到绝对值最小的值，以及其所在的位置。
def read_result(filename,man_made):
    fr=open(filename,'r')
    aline=fr.readlines()
    fr.close()
    min_sub = 100  #时间差最小值
    i       = 0    #行数
    min_i   = 0    #时间差最小所在的行数。
    FP_pick = 0
    FP_pick_list = []
    for line in aline:
        i+=1
        part=line.split()
        s1,s2,s3 = part[6],part[7],part[8]
        newtime  = UTCDateTime(s1+' '+s2+' '+s3) #自动拾取的时间
        subtract = newtime-man_made              #自动与手动拾取的时间差。
        FP_pick_list.append(str(newtime))        #所有的自动拾取结果放在一个列表中
        #找到时间差最小的
        if abs(subtract) < abs(min_sub):
            min_sub = subtract
            min_i   = i
            FP_pick = newtime
    return FP_pick_list
    #return min_sub,min_i,FP_pick #返回的是时间差的绝对值最小值，但是返回的不是绝对值，有正有负，同时返回最小值所在的行数，如果是
                                 #0则表示FP没有拾取，是1则表示第一行的拾取误差最小，因为是i计数是从1开始的。
                                 #newtime是FP拾取的绝对时间
            


'''
#1 第一步：读取test.txt中的数据路径和答案,存储到A中，将数据路径赋值给要调用的程序，得到结果，与标准答案进行对比。
fa = open('test.txt')
A  = fa.readlines()
fa.close()

result = []  #将最好的时间差记下来，里面的内容应该是元组形式的
FP_right_list = [] #拾取误差在0.5s的将文件名保存在这里
for line in A:
    path,answer = line.split()
    if answer != '-1234':  #说明改事件是个地震，而不是噪声
        try:
            subprocess.call('./picker_func_test %s zday1.txt  522 1206 61 10 7' %(path),shell=True) #得到一个数据的结果，检查zday1.txt中的自动拾取的结果。
        
            #计算人工拾取与自动拾取的差
            man_result  = UTCDateTime(answer)  #人工拾取
            ret_result  = read_result('zday1.txt',man_result) #调用函数计算自动拾取与手动拾取的误差最小值
            min_result  = (ret_result[0],ret_result[1])
            result.append(min_result)

            #FP没有拾取到的话，min_sub就会是100(默认是100)，将这些数据挑出来看看，因为有些是因为波形是没有数据的
            if min_result[0]==100:
                os.system('cp %s /home/zhangzhipeng/software/github/2020/data/no_pick_data'%(path))
            
            #FP拾取误差较大的话，即当误差大于1.4s时复制到这里,并且将FP拾取的结果赋值给t1
            elif abs(min_result[0])>1.4:
                os.system('cp %s /home/zhangzhipeng/software/github/2020/data/wrong_pick_data'%(path))
                name = os.path.basename(path)
                st = read('/home/zhangzhipeng/software/github/2020/data/wrong_pick_data/'+name)
                s = st[0].stats.sac
                s.t1 = ret_result[2]-(st[0].stats.starttime-s.b)
                st.write('/home/zhangzhipeng/software/github/2020/data/wrong_pick_data/'+name)
            
            #误差小于0.5的收集起来，生成文件FP_right_list.json,到时与AI生成的结果做一个比对。
            elif abs(min_result[0])<=0.5:
                FP_right_list.append(os.path.basename(path).replace('BHZ.sac','npz'))
                
            #FP生成的的结果文件收集起来，保存到zresult.txt中，同时在其下面添加标准到时。
            fb = open('zday1.txt','r')
            C  = fb.readlines()
            fb.close()
            fc = open('zresult.txt','a+')
            for linec in C:
                fc.write(linec)
            fc.write(line)
            fc.write('\n')
            fc.close()        
            os.system('rm zday1.txt')
            
        #震相报告拾取的到时，有些是错误的，比如2018年发生的地震，到时居然是2016年的，这时候会报错,将其移动到一个位置   
        except ValueError:
            os.system('rm zday1.txt')
            os.system('mv %s /home/zhangzhipeng/software/github/2020/data/wrong_data'%(path))

fb.close()

filename='FP_right_list.json'
with open(filename,'w') as file_obj:
    json.dump(FP_right_list,file_obj)

print (len(result))  #70个地震自动与手动的时间差,以及是第几个地震的时间差最小
#plot_scatter(result,2)      #画散点图，自己看的

right,left = sta_list(result)  #处理统计时间差，将其整理好，以备画图时用，只保留误差在0.13-0.14(1.3)之下的，其他的不要了。
plot_bar(right,left)           #画柱状图
'''



#读取地震事件数据，获得文件路径以及文件名称，获得人工拾取的到时，作为标准答案，形成test.txt,以备后面的对比。




#script 5.1 读取所有的地震事件波形数据，生成test.txt用以check_result.py备用。
'''
fa = open('test.txt','a+')

i = 0
datas = glob.glob('/home/zhangzhipeng/software/github/data/*.BHZ.sac')
for data in datas:
    st = read(data)
    start = st[0].stats.starttime
    at = st[0].stats.sac
    
    
    tp = start+at.a-at.b
    
    #写入数据文件所在路径以及tp到时
    fa.write(data+' '+str(tp))
    fa.write('\n')
    i+=1
fa.close()
print ('there are %s data'%(str(i)))
'''
#6.1 找到2个列表(2种拾取方法结果)的交集，交集可以视为正确的拾取，将不再交集中的三分量数据移动到uncertain文件夹中，并画图，手动挑选。

'''
data_path  = '/home/zhangzhipeng/software/github/2020/data/*.BHZ.sac' #遍历一个月的所有数据 
uncertain_data = '/home/zhangzhipeng/software/github/2020/data/uncertain_right_data/'  #将sac三分量画图后保存位置。


filename1='AI_right_list.json'
with open(filename1) as file_obj:
    AI_right = json.load(file_obj)

filename2 = 'FP_right_list.json'
with open(filename2) as file_obj:
    FP_right = json.load(file_obj)
    
all_right_list = list(set(AI_right)&set(FP_right))



data_files = glob.glob(data_path)       #所有的z分量的数据
for data in data_files:
    name = os.path.basename(data)
    if name.replace('BHZ.sac','npz') not in all_right_list:
        os.system('mv %s %s'%(data.replace('BHZ.sac','*'),uncertain_data))
'''

#6.2 承接上面的6.1，遍历uncertain文件夹中的不确定的数据，然后画图，对画完的图进行手动挑选，将不正确的图删除，剩下的数据移动到原始文件夹中。

'''
def plot_waveform_npz(plot_dir,file_name,data,itp,its): 
    #画sac三分量数据， 将数据格式转换为npz,进行滤波等处理，shape是3,12001，输入画完图的保存路径，文件名称，数据
    plt.figure(figsize=(25,15))
    ax=plt.subplot(data.shape[0],1,1) #(3,1,1) 输入的数据shape是3,12001
    for j in range(data.shape[0]):
        plt.subplot(data.shape[0],1,j+1,sharex=ax)
        t=np.linspace(0,data.shape[1]-1,data.shape[1]) #(0,12000,12001)
        data_max=data[0,:].max()
        data_min=data[0,:].min()
        plt.plot(t,data[j,:])
        plt.vlines(itp,data_min,data_max,colors='r') 
        plt.vlines(its,data_min,data_max,colors='blue') 
        
    plt.suptitle(file_name,fontsize=25,color='r')
    png_name=file_name+'.png' 
    plt.savefig(plot_dir+png_name)
    #os.system('mv *.png png') 
    plt.close()  
    

data_path  = '/home/zhangzhipeng/software/github/2020/data/uncertain_right_data' #遍历一个月的所有数据 
save_png = '/home/zhangzhipeng/software/github/2020/data/uncertain_right_figure/'  #将sac三分量画图后保存位置。

data_files = sorted(glob.glob(data_path+'/*BHZ.sac'))       #一个月的所有天数
for datas in data_files:
    #遍历所有的z分量数据，并以此找到三分量数据
    file_name   = os.path.basename(datas)
    figure_name = file_name.replace('BHZ.sac','')
    
    st = read(datas.replace('BHZ.sac','*'))     
    co=st.copy()
    #去均值，线性，波形歼灭,然后滤波
    co.detrend('demean').detrend('linear').taper(max_percentage=0.05, max_length=10.)
    co=co.filter('bandpass',freqmin=1,freqmax=15) #带通滤波
    
    #将滤波后的数据转换成numpy格式，
    data=np.asarray(co)

    tp = st[0].stats.sac.a
    b  = st[0].stats.sac.b
    tp_num = int((tp-b)*100) #p波到时的点数是P波到时减去b值，参考时刻是发震时刻，p波到时的点数应该是1500，因为截取的时候是前15s
    ts_num = 11000
    
    try:
        plot_waveform_npz(save_png,figure_name,data,tp_num,ts_num)
    except IndexError:
        print (datas)

'''
#6.3 承接上面的6.2，遍历挑选后剩下的图，获得图名，然后按照图名将uncertain_right_data中的数据移动到原始data文件夹中。

'''
png_path  = '/home/zhangzhipeng/software/github/2020/data/uncertain_right_figure/*.png'  #将sac三分量画图后保存位置。
data_path = '/home/zhangzhipeng/software/github/2020/data/' 

png_files = glob.glob(png_path)

for png in png_files:
    data_name = png.replace('.png','*').replace('uncertain_right_figure','uncertain_right_data')
    os.system('mv %s %s'%(data_name,data_path))
'''


#6.4 承接上面的6.3 遍历经过挑选的数据，生成相应的误差分布图。

'''
fa = open('test.txt')
A  = fa.readlines()
fa.close()

result = {}  #字典，键是文件名称，值是FP算法对应的拾取(str格式，可以转换为UTC格式的时间)
for line in A:
    path,answer = line.split()
    sac_name = os.path.basename(path)
    if answer != '-1234':  #说明改事件是个地震，而不是噪声
        try:
            subprocess.call('./picker_func_test %s zday1.txt  522 1206 61 10 7' %(path),shell=True) #得到一个数据的结果，检查zday1.txt中的自动拾取的结果。
        
            #计算人工拾取与自动拾取的差
            man_result  = UTCDateTime(answer)  #人工拾取
            ret_result  = read_result('zday1.txt',man_result) #调用函数计算自动拾取与手动拾取的误差最小值
            
            result.setdefault(sac_name,[]).extend(ret_result)
    
            os.system('rm zday1.txt')
        #震相报告拾取的到时，有些是错误的，比如2018年发生的地震，到时居然是2016年的，这时候会报错,将其移动到一个位置   
        except ValueError:
            os.system('rm zday1.txt')
            os.system('mv %s /home/zhangzhipeng/software/github/2020/data/wrong_data'%(path))



keys = result.keys()
print (len(keys))


filename='fp_pick.json'
with open(filename,'w') as file_obj:
    json.dump(result,file_obj)
'''

'''
#原始数据所在的位置
data_path = '/home/zhangzhipeng/software/github/data'
sac_files = glob.glob(data_path+'/*.BHZ.sac')
save_path = '/home/zhangzhipeng/software/github/data/no_pick'
save_wrong = '/home/zhangzhipeng/software/github/data/wrong_pick'

#FP对上述原始数据拾取结果
filename='fp_pick.json'
with open(filename) as file_obj:
    fp_pick = json.load(file_obj)

keys = fp_pick.keys()
print(len(keys))

no_pick,one_pick,two_pick,three_pick = 0,0,0,0
os.chdir(data_path)
#遍历所有的原始数据，将没有和错误的拾取分别放在不同的位置，并且对错误的拾取添加t9,t8等。
for sac_file in sac_files:
    sac_name = os.path.basename(sac_file)
    if sac_name in keys: #keys中对应的值是fp拾取的结果，有的有0个拾取，有的可能有多个拾取
        value = fp_pick[sac_name]
        if len(value)==0:
            no_pick+=1
            os.system('cp %s %s'%(sac_name,save_path))
        else:    
            st = read(sac_name)
            c_tr = st[0].copy()
            s = c_tr.stats.sac
            if len(value)==1:    
                s.t9 = UTCDateTime(value[0])-c_tr.stats.starttime
                one_pick +=1
            elif len(value)==2:
                s.t9 = UTCDateTime(value[0])-c_tr.stats.starttime
                s.t8 = UTCDateTime(value[1])-c_tr.stats.starttime
                two_pick +=1
            elif len(value)>=3:
                s.t9 = UTCDateTime(value[0])-c_tr.stats.starttime
                s.t8 = UTCDateTime(value[1])-c_tr.stats.starttime
                s.t7 = UTCDateTime(value[2])-c_tr.stats.starttime
                three_pick +=1
            data_name = save_wrong+'/'+sac_name
            c_tr.write(data_name,format='SAC')

print('no_pick %s one_pick %s two_pick %s three_pick and above %s'%(no_pick,one_pick,two_pick,three_pick))
'''


#1.5 sac单分量数据画图， 将数据格式转换为npz,进行滤波等处理，shape是1,9001，输入画完图的保存路径，文件名称，数据，tp到时

def plot_waveform_npz(plot_dir,file_name,data,itp): 
    plt.figure(figsize=(25,15))
    data = data
    t=np.linspace(0,data.shape[1]-1,data.shape[1]) #(0,9000,9001)
    plt.plot(t,data[0,:])        
    data_max=data.max()
    data_min=data.min()
    tp_num = itp
    plt.vlines(tp_num[0],data_min,data_max,colors='r') 
    #plt.vlines(tp_num[1],data_min,data_max,colors='r') 
    
    #title = str(tp_num[0])+'-'+str(tp_num[1])
    title = str(tp_num[0])
    plt.suptitle(title,fontsize=25)
    
    png_name=plot_dir+'/'+file_name+'png' #保留的文件名是信噪比加后面的信息
    plt.savefig(png_name)
    plt.close()  


data_path  = '/home/zhangzhipeng/software/github/data/no_pick' #遍历一个月的所有数据 
save_dir = '/home/zhangzhipeng/software/github/data/no_pick/figure'  #将sac三分量画图后保存位置。

data_files = sorted(glob.glob(data_path+'/*BHZ.sac'))       #一个月的所有天数
total = len(data_files)
num =0
for data in data_files:
    #遍历所有的z分量数据，并以此找到三分量数据
    file_name   = os.path.basename(data)
    figure_name = file_name.replace('BHZ.sac','')
    
    st = read(data)     
    co=st.copy()
    #去均值，线性，波形歼灭,然后滤波
    co.detrend('demean').detrend('linear').taper(max_percentage=0.05, max_length=10.)
    co=co.filter('bandpass',freqmin=1,freqmax=15) #带通滤波
    
    #将滤波后的数据转换成numpy格式，
    data=np.asarray(co)
    tp_list = [3000]
    
    '''
    try:
        tp1 = st[0].stats.sac.t9
        tp2 = st[0].stats.sac.t8
        tp_list = [tp1,tp2]
    except AttributeError:
        tp_list = [tp1,60]
    '''  
    #tp_list = [int(i*100) for i in tp_list]
    plot_waveform_npz(save_dir,figure_name,data,tp_list)













#6.5 遍历已经完全挑选好的数据中的z分量，从中截取其中的前n秒作为噪声，做测试

'''
data_path  = '/home/zhangzhipeng/software/github/2020/data/*.BHZ.sac' #遍历挑选好的所有数据 
noise_path = '/home/zhangzhipeng/software/github/2020/data/noise_data/'  #
sac_files = glob.glob(data_path)
for sac in sac_files:
    sac_name = os.path.basename(sac)
    st = read(sac)
    st.sort(keys=['channel'], reverse=False) #对三分量数据排序
    start = st[0].stats.starttime
    
    c_tr = st.copy()
    data = c_tr.trim(start, start+60,pad=True, fill_value=0) 
    
    noise_name = noise_path+sac_name
    data.write(noise_name,format='SAC')

    print (noise_name,data)
'''

#6.6 遍历noise_data中的数据，画图，大概看一下
'''
data_path  = '/home/zhangzhipeng/software/github/2020/data/noise_data/*.BHZ.sac' #遍历挑选好的所有数据 
noise_fig  = '/home/zhangzhipeng/software/github/2020/data/noise_figure/'  #

noise_files = glob.glob(data_path)

for noise in noise_files:

    name = os.path.basename(noise)
    st = read(noise) 
    save_name = noise_fig+name.replace('BHZ.sac','png')  
    st.plot(equal_scale=False,outfile=save_name,color='red',size=(1800,1000)) 
'''


#6.7 读取所有的噪声事件波形数据，生成test.txt用以check_result.py备用。

'''
fa = open('test.txt','a+')

i = 0
datas = glob.glob('/home/zhangzhipeng/software/github/2020/data/noise_data/*.BHZ.sac')
for data in datas:
    #写入数据文件所在路径以及tp到时
    fa.write(data+' '+'-1234')
    fa.write('\n')
    i+=1
fa.close()
print ('there are %s data'%(str(i)))


#根据text.txt中的文件列表，用FP将数据遍历一遍，读取生成的zday1.txt，看其是否是空，空的话说明没有拾取到，对于噪声来说就是正常的，
#最好显示一共多少个噪声，有几个拾取到，几个没有
fa = open('test.txt')
A  = fa.readlines()
fa.close()

total = len(A) #总的噪声的数量
pick_num = 0   #拾取的数量，因为是噪声，拾取说明是错误。


for line in A:
    path,answer = line.split()
    if answer == '-1234':  #说明改事件是个地震，而不是噪声
        try:
            subprocess.call('./picker_func_test %s zday1.txt  522 1206 61 10 7' %(path),shell=True) #得到一个数据的结果，检查zday1.txt中的自动拾取的结果。
        
            fb = open('zday1.txt','r')
            B  = fb.readlines()
            fb.close()
            if len(B)!=0:
                pick_num+=1 
            os.system('rm zday1.txt')
            
        except Exception as e: #所有异常，输出到文件中
            print (e)
            
print ('there are %s noise, FP picked %s which is wrong'%(total,pick_num))

'''






















































