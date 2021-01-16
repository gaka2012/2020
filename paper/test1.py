#!/usr/bin/env python
#-*- coding:utf-8 -*-

from obspy.core import UTCDateTime
import csv,sys,json,os,glob,re
import numpy as np
from obspy.core import read
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import entropy as sci_entropy
from itertools import groupby
from matplotlib import font_manager

font = font_manager.FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")

def plot_multi_bar(y1_list,y2_list,y3_list):
    name_list = ['0-50','50-100','100-150','>150']  #x轴坐标名称
    y_name_list = ['5%','10%','15%','20%'] #y轴坐标名称
    #y1_list = [1.5,0.6,7.8,6]  #最左边柱状图纵坐标  
    #y2_list = [1,2,3,1]        #最右边柱状图纵坐标
    #y3_list = [2,2,2,2]        #中间柱状图纵坐标

    a =list(range(len(y1_list))) #x坐标  
    x = np.array(a)

    total_width, n = 0.8, 4  
    width = 0.15  
    
    plt.figure(figsize=(20,12),dpi=200)
    
    plt.bar(x-width/2, y1_list, width=width, label='FilterPicker',fc = 'red')  

    plt.bar(x+width/2, y2_list, width=width, label='PhaseNet',fc = 'dodgerblue')  

    x_label = [0,1,2,3] #在哪个位置写x轴标签
    y_label = [0.05,0.1,0.15,0.2]

    plt.xticks(x_label,name_list,fontsize=30)  #写x轴标签,x_label坐标同时也是x轴的锚点，如果没有后面的name_list，则显示的x轴坐标是x_label的内容
    plt.yticks(y_label,y_name_list,fontsize=30)
    plt.xlabel('信噪比',fontproperties=font,fontsize=35) #加上横纵坐标轴的描述。
    plt.ylabel('错误率',fontproperties=font,fontsize=35) #加上横纵坐标轴的描述。
    
    plt.legend(loc='upper right', frameon=True,fontsize=20)  
    plt.savefig('multiply_bar')
    #plt.show()
    
    
#将输入信噪比列表分成4类，返回每一类对应的数量
def return_snr_dist(snr_list): 
    sum_num = 0 #大于等于3的数据累加
    fp_snr = []  #fp拾取的地震事件的信噪比，是一个列表，有4组，分别代表信噪比为 ['0-50','50-100','100-150','>150']  
    for k,g in groupby(sorted(snr_list),key=lambda x:x//50): #统计num_list中每个数字的个数。
        num = len(list(g)) 
        if k>=3:
            k = 3
            sum_num+=num
        else:
            fp_snr.append(num)
    fp_snr.append(sum_num)
    return fp_snr


def plot_line_chart(y,y1):

    #在同一幅图中，画2个散点图
    plt.figure(figsize=(20,12),dpi=600)

    ax = plt.subplot(1,1,1)
    x = [i for i in range(len(y))] #根据能量的长度生成x轴。
    #plt.plot(y,marker='o',label='event',color='red')
    plt.scatter(x,y,c='red',s=6,marker='o') #c是颜色，s是画的点的大小
    #图二：噪声熵值折线图    
    
    #plt.plot(y1,label='noise')
    x1 = [i for i in range(len(y1))] #根据能量的长度生成x轴。
    plt.scatter(x1,y1,c='blue',s=6,label='PhaseNet') #c是颜色，s是画的点的大小
    

    
    plt.hlines(2.5,0,2617,colors='black') 
    
    #name = 'event_entropy_line_chart'
    #plt.title(name,fontsize=24,color='r')
    plt.xlabel('x',fontsize=25) #加上横纵坐标轴的描述。
    plt.ylabel('y',fontsize=25)

    #x_show = [0,10,20,30,40,50,60,70,80,90,100]
    x_show = [500,1000,1500,2000,2500]

    y_show = [0,0.5,1.0,1.5,2.0,2.5,3.0]
    ax.set_xticks(x_show)             #显示的坐标，只显示0,1,10,其他不显示
    ax.set_xticklabels(x_show,rotation=0,fontsize=30)  #
    ax.set_yticks(y_show)             #显示的坐标，只显示0,1,10,其他不显示
    ax.set_yticklabels(y_show,rotation=0,fontsize=30)  #


    #将图片保存下来。
    #plt.legend(loc=0,prop={'size':30})
    plt.xlabel('事件',fontproperties=font,fontsize=40) #加上横纵坐标轴的描述。
    plt.ylabel('熵值',fontproperties=font,fontsize=40) #加上横纵坐标轴的描述。
    plt.savefig('event_noise_entropy.png')
    plt.close()


def short_term_energy(chunk):
    return np.sum((np.abs(chunk) ** 2) / chunk.shape[0])

def energy_per_frame(windows):
    out = []
    for row in windows: #row代表每一行
        out.append(short_term_energy(row)) #每个值的平方和再处以5
    return np.hstack(np.asarray(out))


#子函数，计算熵值，首先将数据转换成numpy格式，然后作预处理，
def calculate_entropy(tr,tp_num):
    
    preceding_time,duration_time = 5,15 #p波到时前5s，总长度为15s
    num = tp_num
    fm = 100 #默认频率是100，秒数乘以频率得到点数。
    if num+duration_time*100>=9000:
        num = 9000-duration_time*100
    elif num-preceding_time<=0:
        num = preceding_time*100
    
    #计算实际地震熵值
    event = data[num-preceding_time*fm:num-preceding_time*fm+duration_time*fm]  #15秒长的数据    
    event_part   = np.reshape(event,[-1,100])       #reshape一下，将其转换为n(15)个长度为100的片段。
    
    event_energy = energy_per_frame(event_part)  #得到15个片段的能量值
    event_out    = sci_entropy(event_energy) #求熵值,即15个平均能力的熵值。
    return round(event_out,2)



def plot_waveform_npz(plot_dir,file_name,data,itp,entropy): 
    plt.figure(figsize=(25,15))
    data = data
    t=np.linspace(0,data.shape[0]-1,data.shape[0]) #(0,9000,9001)
    plt.plot(t,data)        
    data_max=data.max()
    data_min=data.min()
    tp_num = itp
    plt.vlines(tp_num,data_min,data_max,colors='r') 
    #plt.vlines(tp_num[1],data_min,data_max,colors='r') 
    
    title = str(entropy)
    plt.suptitle(title,fontsize=25)
    
    png_name=plot_dir+'/'+file_name+'png' #保留的文件名是信噪比加后面的信息
    plt.savefig(png_name)
    plt.close()  



#1.1遍历FP拾取的噪声数据：画图计算其熵值(已经将其中的地震事件剔除，只剩下纯噪声数据)结果保存在json文件中。

'''
filename = 'noise_wrong.json'
with open(filename) as file_obj:
    info  = json.load(file_obj)    

keys = list(info.keys())

data_path = '/home/zhangzhipeng/software/data/noise_data/fp_pick' #存放噪声数据的位置
save_png  = '/home/zhangzhipeng/software/data/noise_data/fp_pick/figure'
datas = glob.glob(data_path+'/*.BHZ.sac')
datas = [os.path.basename(i) for i in datas]

num = 0
os.chdir(data_path)
noise = {} #噪声的熵值
for sac_name in keys:
    if sac_name in datas:
        num+=1
        st = read(sac_name)
        start = st[0].stats.starttime
        end = st[0].stats.endtime
        pick = UTCDateTime(info[sac_name][0])
        tp = int((pick-start)*100)
        #print(tp)
    
        co=st[0].copy()
        #去均值，线性，波形歼灭,然后滤波
        co.detrend('demean').detrend('linear').taper(max_percentage=0.05, max_length=10.)
        co=co.filter('bandpass',freqmin=1,freqmax=15) #带通滤波
            
        #将滤波后的数据转换成numpy格式，
        data=np.asarray(co)         
        entropy = calculate_entropy(data,tp)
        #print (entropy)
        temp = [entropy]
        noise.setdefault(sac_name,[]).extend(temp)
        #plot_waveform_npz(save_png,sac_name.replace('BHZ.sac',''),data,tp,entropy)
print('there are %s noise'%(num))

with open('/home/zhangzhipeng/software/github/2020/fp_noise_entropy.json','w') as ob:
    json.dump(noise,ob)
'''

#1.2 读取FP拾取的噪声数据熵值形成的列表
'''
with open('fp_noise_entropy.json') as ob:
    fp_noise_entropy = json.load(ob)
print('there are %s noise'%(len(fp_noise_entropy)))

#统计噪声数据熵值中大于某个熵值的数据
threshold = 2.3
big_thre = list(filter(lambda a:a>threshold,fp_noise_entropy))


print ('%s noise entropy bigger than %s'%(len(big_thre),threshold))
'''


'''
data_path = '/home/zhangzhipeng/software/data/mag_three_data' #存放噪声数据的位置
event = []
num = 0
datas = glob.glob(data_path+'/*.BHZ.sac')
for sac_name in datas:
    num+=1
    tp = 3000
    st = read(sac_name)
    co=st[0].copy()
    #去均值，线性，波形歼灭,然后滤波
    co.detrend('demean').detrend('linear').taper(max_percentage=0.05, max_length=10.)
    co=co.filter('bandpass',freqmin=1,freqmax=15) #带通滤波
            
    #将滤波后的数据转换成numpy格式，
    data=np.asarray(co)         
    entropy = calculate_entropy(data,tp)
    event.append(entropy)

print('there are %s event'%(num))

os.chdir('/home/zhangzhipeng/software/github/2020')


filename = 'mag_two_entropy.json'
with open(filename) as file_obj:
    ai_noise  = json.load(file_obj)    

plot_line_chart(noise,event,ai_noise)
'''


'''
#将待删除的图片的名称文件删除。
png_path = '/home/zhangzhipeng/software/data/noise_data/fp_eq/*.png'
data_path = '/home/zhangzhipeng/software/data/noise_data/fp_pick'
delete_sac = glob.glob(png_path)
os.chdir(data_path)
for sac in delete_sac:
    sac_name = os.path.basename(sac).replace('png','BHZ.sac')
    os.system('rm %s'%(sac_name))
    
    
print('there are %s files need to be removed'%(len(delete_sac)))

'''

'''
/bashi_fs/centos_data/zzp/120_data

/bashi_fs/centos_data/zzp/120_data/wrong_data
'''

#将2级以上的地震移动到mag_path

'''
data_path = '/home/zhangzhipeng/software/data/9989_data' #存放原始数据的位置
save_path = '/home/zhangzhipeng/software/data/mag_two_data'

filename = '90_s_event_info.json'
with open(filename) as file_obj:
    info  = json.load(file_obj)    

#keys = list(info.keys())

os.chdir(data_path)
num = 0
for key,value in info.items():
    mag,dist = float(value[1]),float(value[0])
    if mag >0 :
        num +=1
        print(mag,key,dist)
        os.system('cp %s %s'%(key,save_path))
print (num)




#计算0级以上震中距小于50km地震熵值，并将熵值大于2.2的放到一个文件夹中

data_path = '/home/zhangzhipeng/software/data/mag_two_data' #存放2级以上地震数据的位置
save_png  = '/home/zhangzhipeng/software/data/mag_two_data/figure'
event_entropy = {}
num = 0
datas = glob.glob(data_path+'/*.BHZ.sac')
for sac_name in datas:
    name = os.path.basename(sac_name)
    tp = 3000
    st = read(sac_name)
    co=st[0].copy()
    #去均值，线性，波形歼灭,然后滤波
    co.detrend('demean').detrend('linear').taper(max_percentage=0.05, max_length=10.)
    co=co.filter('bandpass',freqmin=1,freqmax=15) #带通滤波
            
    #将滤波后的数据转换成numpy格式，
    data=np.asarray(co)         
    entropy = calculate_entropy(data,tp)
    temp = [entropy]
    event_entropy.setdefault(name,[]).extend(temp)
    if entropy >2.5:
        #plot_waveform_npz(save_png,name.replace('BHZ.sac',''),data,tp,entropy)
        num+=1
with open('/home/zhangzhipeng/software/github/2020/event_entropy.json','w') as ob:
    json.dump(event_entropy,ob)    
    
print (num)

'''

'''
filename='mag_two_entropy.json'
with open(filename,'w') as file_obj:
    json.dump(event,file_obj)
'''





#将ai待删除的图片的名称文件删除,data_path原本有2065个噪声数据，移动后只剩下1705个数据
'''
png_path = '/home/zhangzhipeng/software/data/noise_data/ai_pick/figure/*.png'  #ai拾取的噪声经过手动挑选后剩下的图
data_path = '/home/zhangzhipeng/software/data/noise_data/ai_pick'              #ai拾取的所有噪声
save_data = '/home/zhangzhipeng/software/data/noise_data/ai_pick/event_figure/data'  #将拾取的噪声中的地震移动到这个位置

pngs  = glob.glob(png_path)  #
pngs  = [os.path.basename(i).replace('.png','') for i in pngs] #SC.WCH_20181117100136.mseed

os.chdir(data_path)
datas =  glob.glob(data_path+'/*.mseed')
for data in datas:
    mseed_name = os.path.basename(data)
    if mseed_name in pngs:
        pass
    else:
        os.system('mv %s %s'%(mseed_name,save_data))
'''
      
#计算ai错误拾取的噪声的熵值，生成字典，键是文件名，值是对应的熵值。
#读取的json中是ai错误拾取的噪声数据名称以及对应的拾取，

'''
filename='./PhaseNet-master/ai_noise.json'
with open(filename) as file_obj:
    ai_noise = json.load(file_obj)
      
mseed_path = '/home/zhangzhipeng/software/data/noise_data/ai_pick' #将ai错误拾取的noise所在路径，剩下1705个
mseeds = glob.glob(mseed_path+'/*.mseed')

noise_entropy = {}
num = 0
for mseed in mseeds:
    mseed_name = os.path.basename(mseed)
    if mseed_name in ai_noise.keys():
        tp = ai_noise[mseed_name]
        st = read(mseed)    
        num+=1
        st.sort(keys=['channel'], reverse=False) #对三分量数据排序
    
        #只计算z分量的熵值
        co=st[2].copy()
        #去均值，线性，波形歼灭,然后滤波
        co.detrend('demean').detrend('linear').taper(max_percentage=0.05, max_length=10.)
        co=co.filter('bandpass',freqmin=1,freqmax=15) #带通滤波
            
        #将滤波后的数据转换成numpy格式，
        data=np.asarray(co)    #(9001,)   
        
        try:
            entropy = calculate_entropy(data,tp)
            temp = [entropy]
            noise_entropy.setdefault(mseed_name,[]).extend(temp)
                
            #plot_waveform_npz(mseed_path+'/figure/',mseed_name,data,tp,entropy)
        except ValueError:
            print(mseed_name)
print('there are %s noise'%(num))     

      
with open('/home/zhangzhipeng/software/github/2020/ai_noise_entropy.json','w') as ob:
    json.dump(noise_entropy,ob)

'''

#统计fp拾取噪声数据熵值中大于某个熵值的数据
'''
with open('fp_noise_entropy.json') as ob:
    fp_noise_entropy = json.load(ob)
    
fp_noise_num = []
for key,value in fp_noise_entropy.items():
    fp_noise_num.append(value[0])

print('fp picked %s noise'%(len(fp_noise_num)))

threshold = 2.5
big_thre = list(filter(lambda a:a>threshold,fp_noise_num))

print ('%s noise entropy bigger than %s'%(len(big_thre),threshold))


#统计ai拾取噪声数据熵值中大于某个熵值的数据
with open('ai_noise_entropy.json') as ob:
    ai_noise_entropy = json.load(ob)
    
noise_num = []
for key,value in ai_noise_entropy.items():
    noise_num.append(value[0])

print('ai picked %s noise'%(len(noise_num)))

threshold = 2.5
big_thre = list(filter(lambda a:a>threshold,noise_num))


print ('%s noise entropy bigger than %s'%(len(big_thre),threshold))

'''



#看一下2种方法拾取的同一个文件中的噪声的值是否相同，(不一定，因为拾取的位置不一样)
'''
with open('fp_noise_entropy.json') as ob:
    fp_noise_entropy = json.load(ob)
fp_noise = list(fp_noise_entropy.keys())
print('there are %s noise'%(len(fp_noise)))


with open('ai_noise_entropy.json') as ob:
    ai_noise_entropy = json.load(ob)
noise_num = list(ai_noise_entropy.keys())
print('there are %s noise'%(len(noise_num)))

num = 0
for ai in noise_num:
    ai_name = ai.replace('mseed','BHZ.sac')
    if ai_name in fp_noise:
        print (ai_name,ai_noise_entropy[ai],fp_noise_entropy[ai_name])
        num+=1
        
print (num)
'''


#绘制简单饼状图
'''
# Pie chart, where the slices will be ordered and plotted counter-clockwise:
labels = 'Frogs', 'Hogs', 'Dogs', 'Logs'
sizes = [15, 30, 45, 60]
explode = (0, 0, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')
colors = ['white','white','white','white']
 
fig1, ax1 = plt.subplots()

ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=False, startangle=90,colors=colors,wedgeprops={'linewidth':0.5,'edgecolor':'black'})
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
 
 
#plt.show()
plt.savefig('test')
'''


#计算所有地震事件的信噪比,存储在name_snr.json，键是文件名，值是对应的信噪比
'''
data_path = '/home/zhangzhipeng/software/data/9989_data'

name_snr = {}
num = 0
datas = glob.glob(data_path+'/*.BHZ.sac')
total = len(datas)

for sac in datas:
    num+=1
    percent=num/total #用于写进度条
    sys.stdout.write("\r{0}{1}".format("#"*10 , '%.2f%%' % (percent * 100)))
    sys.stdout.flush()
    name = os.path.basename(sac)
    st = read(sac)
    #只计算z分量的熵值
    co=st[0].copy()
    #去均值，线性，波形歼灭,然后滤波
    co.detrend('demean').detrend('linear').taper(max_percentage=0.05, max_length=10.)
    co=co.filter('bandpass',freqmin=1,freqmax=15) #带通滤波
            
    #将滤波后的数据转换成numpy格式，
    data=np.asarray(co)    #(9001,)   
    tp  = 3000
    event = np.var(data[tp:tp+500])
    noise = np.var(data[tp-500:tp])
    snr   = event/noise
    snr   = round(snr,2)
    temp  = [snr]

    name_snr.setdefault(name,[]).extend(temp)
    

with open('name_snr.json','w') as ob:
    json.dump(name_snr,ob)
'''

#读取所有的信噪比，画信噪比饼状图
'''
with open('name_snr.json') as ob:
    name_snr = json.load(ob)
snr = []
for key,value in name_snr.items():
    snr.append(value[0])
    
print ('there are %s snr'%(len(snr)))


#对信噪比进行分类
snr_dict = {}
for k,g in groupby(sorted(snr),key=lambda x:x//50): #统计num_list中每个数字的个数。
    #print (k,len(list(g)))
    num = len(list(g)) 
    if k>=3:
        k = 3
    snr_dict.setdefault(k,[]).extend([num])
    
snr_dict[3] = [sum(snr_dict[3])]

#计算每个类别信噪比所占比值,用于后面的画图
sizes = []
for key,value in snr_dict.items():
    value = value[0]/len(snr)
    value = round(value,2)
    sizes.append(value)
print (sizes)


#绘制简单饼状图
# Pie chart, where the slices will be ordered and plotted counter-clockwise:
labels = '0-50', '50-100', '100-150', '>150'
explode = (0, 0, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')
#colors = ['white','white','white','white']
colors = ['lightgreen','gold','lightskyblue','lightcoral']
 
fig1, ax1 = plt.subplots()

ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=False, startangle=90,colors=colors,wedgeprops={'linewidth':0.5,'edgecolor':'black'})
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
 
 
#plt.show()
plt.savefig('snr_dist')
'''



'''
filename = '90_s_event_info.json'
with open(filename) as file_obj:
    info  = json.load(file_obj)    

#keys = list(info.keys())

num = 0
for key,value in info.items():
    mag,dist = float(value[1]),float(value[0])
    if mag >0 and dist>250:
        num +=1
        print(mag,key,dist)
print (num)
'''


#4类信噪比情况下，2种算法的错误率
#打开存储信噪比的文件以及FP拾取地震事件的结果，统计错误拾取的事件的信噪比，存储在列表wrong_snr中
'''
with open('name_snr.json') as ob:
    name_snr = json.load(ob)
    
print ('there are %s snr'%(len(name_snr)))

snr = list(name_snr.values())
snr = [i[0] for i in snr]  #所有数据的信噪比


#打开fp拾取结果的文件，统计错误拾取的文件的信噪比
with open('Plot_result.json') as ob:
    fp = json.load(ob)

wrong_snr = []
wrong_num =0
for key,value in fp.items():
    if value[0]==0:
        wrong_num+=1
        if key in name_snr.keys():
            wrong_snr.append(name_snr[key][0])

print ('fp has %s wrong picks'%(wrong_num))
print ('there are %s wrong picks snr'%(len(wrong_snr)))

wrong_snr = sorted(wrong_snr)  #fp错误拾取的地震事件的信噪比



#打开ai拾取结果的文件，统计错误拾取的文件的信噪比,键是文件名，值是自动拾取与手动拾取的误差点数
with open('ai_name_pick.json') as ob:
    ai = json.load(ob)

ai_snr = []
for key,value in name_snr.items():
    name = key.replace('BHZ.sac','mseed')
    if name in ai.keys():
        resudal = abs(ai[name])
        if resudal > 50:
            ai_snr.append(name_snr[key][0])
        
    else:
        ai_snr.append(name_snr[key][0])


print ('ai picks  %s wrong picks snr'%(len(ai_snr)))


#得到所有地震事件、fp错误拾取、ai错误拾取的事件的信噪比，分为4类，所以得到3个列表，每个列表有4个数，代表4个类别对应的数量
all_snr = return_snr_dist(snr)     
fp_snr  = return_snr_dist(wrong_snr)
ai_snr  = return_snr_dist(ai_snr)
print (all_snr,fp_snr,ai_snr)
fp_snr = list(np.array(fp_snr)/np.array(all_snr))
ai_snr = list(np.array(ai_snr)/np.array(all_snr))
plot_multi_bar(ai_snr,fp_snr,all_snr)
'''

  
#得到fp拾取的噪声的熵值，ai拾取的噪声的熵值，地震事件中震级在0以上，震中距50km以内的地震的熵值，画折线图
#折线图中分成2部分数据，ai拾取错误和fp拾取的噪声合并为一个列表，地震事件单独一个列表
'''
with open('fp_noise_entropy.json') as ob:
    fp_noise_entropy = json.load(ob)
    
fp_noise_num = []
for key,value in fp_noise_entropy.items():
    fp_noise_num.append(value[0])

with open('ai_noise_entropy.json') as ob:
    ai_noise_entropy = json.load(ob)
    
    
ai_noise = list(ai_noise_entropy.values())
ai_noise = [i[0] for i in ai_noise]

ai_fp = []
ai_fp.extend(fp_noise_num)
ai_fp.extend(ai_noise)

print('fp picks %s noise and AI picks %s noise'%(len(fp_noise_num),len(ai_noise)))    
    
with open('event_entropy.json') as ob:
    name_snr = json.load(ob)
event_entropy = list(name_snr.values())
event_entropy = [i[0] for i in event_entropy]
part_event = event_entropy[:2617]
#print (part_event)
  
threshold = 2.5
big_thre = list(filter(lambda a:a>threshold,part_event))



fp_big = list(filter(lambda a:a>threshold,fp_noise_num))
ai_big = list(filter(lambda a:a>threshold,ai_noise))


print ('event has %s and we choose %s event for plotting %s of event entropy bigger than threshold %s'%(len(event_entropy),
        len(part_event),len(big_thre),threshold))
print('fp noise has %s entropy bigger than %s and ai has %s'%(len(fp_big),threshold,len(ai_big)))

plot_line_chart(ai_fp,part_event)
'''


