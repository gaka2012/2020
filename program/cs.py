#!/usr/bin/python
# -*- coding:UTF-8 -*-

from obspy.core import UTCDateTime
import csv
import numpy as np
import time,os
import subprocess
from obspy.core import read
import glob,re
import matplotlib.pyplot as plt
from itertools import groupby

'''
st = read('/home/zhangzhipeng/software/github/2020/program/sac_data/*.sac')
st.sort(['starttime'])
for tr in st:
    s = tr.stats.sac
    r_time = UTCDateTime(year=s.nzyear,julday=s.nzjday,hour=s.nzhour,minute=s.nzmin,second=s.nzsec,microsecond=s.nzmsec*1000) #参考时刻
    p_time = r_time+s.a
    print (tr.stats.starttime,r_time,p_time,s.a,s.evla,s.dist,s.mag,s.az) #输出绝对时间、参考时间、p到时

'''




#subprocess.Popen(['./new','2','2','3'])
#p = subprocess.Popen('./new 2 2 3',shell=True,stdout=subprocess.PIPE)
#out =p.stdout.readlines()
#print ('out == ',out)  


'''
st=read('2019.168.14.52.59.0000.SC.HMS.00.BHZ.D.SAC')
tr=st[0]
co=tr.copy()
#去均值，线性，波形歼灭,然后滤波
co.detrend('demean').detrend('linear').taper(max_percentage=0.05, max_length=10.)
co=co.filter('bandpass',freqmin=1,freqmax=15) #带通滤波
co.filter('highpass',freq=1)#高通滤波
#将滤波后的数据转换成numpy格式
data=np.asarray(co)
'''

'''
st = read('SC.AXI_20180101062906.BHZ.sac')
h0 = st[0].stats.starttime
h1 = st[0].stats.endtime
png_file = 'SC.AXI_20180101062906.BHZ.png'
st.plot(starttime= h0 ,endtime=h1,equal_scale=False,outfile=png_file,color='red',size=(1800,1000)) #单张图，这个size比较好，三张图一起画，默认size就行。
'''



#1.3 读取截取好的地震事件，画震级分布图
def plot_num_distribution(num_list,fig_name): #画不同震级的数量分布图.输入不同的震级形成的列表,处数，震级最大值,横纵坐标轴名
    
    #统计num_list中每个震级的数量,形成2个列表，一个震级，一个是震级对应的数量
    x_list,y_list = [],[] 
    for mag,num in groupby(sorted(num_list),key=lambda x:x): 
        print (mag)
        x_list.append(mag)
        y_list.append(len(list(num)))
    #x,y坐标轴显示的坐标
    new_x = [0,1,2,3,4,5,6,7,8]
    new_y = [0,20,40,60,80,100]
    
    #画图
    plt.figure(figsize=(25, 15))
    plt.bar(x_list,y_list,color='gray',width=0.1,ec='black') #柱状图的宽度,描边
    plt.xticks(new_x,fontsize=30)
    plt.yticks(new_y,fontsize=30)
    plt.xlabel('mag Ms',fontsize=30) #加上横纵坐标轴的描述。
    plt.ylabel('count',fontsize=30) #
    
    plt.savefig(fig_name)  #注意要在plt.show之前
    #plt.close()


#遍历数据形成字典，
'''
data_files = glob.glob('/home/zhangzhipeng/software/github/2020/data/*.BHZ.sac')
name_dict = {}
for data_file in data_files[:10]:
    data_name = os.path.basename(data_file)
    st = read(data_file)
    at = st[0].stats.sac
    
    dist = at.dist
    dist_list = [int(dist)//50]

    name_dict.setdefault(data_name,[]).extend(dist_list)

for key,value in name_dict.items():
    print (key,value)    
'''

#sac文件中的详细信息，需要传入trace
class Sac_info():
    def __init__(self,tr):
        self.info = tr.stats.sac
        self.dist = self.info.dist
        self.mag  = self.info.mag
        
class Noise_process():
    def __init__(self,tr,noise_path,sac_name):
        if not os.path.exists(noise_path):
            os.makedirs(noise_path)
        data_starttime = tr.stats.starttime
        c_tr = tr.copy()
        data = c_tr.trim(data_starttime, data_starttime+90,pad=True, fill_value=0)     
        noise_name = noise_path+sac_name
        data.write(noise_name,format='SAC')


class Sac_files():
    def __init__(self,sac_file_path,get_noise=False):
        self.sac_file_path = sac_file_path
        self.sac_info_all = {}  #存储sac头文件的所有信息  
        self.get_noise = get_noise #是否截取噪声


        self.build() #实例化类直接运行这个函数,最好放在最后，让前面的属性先赋值完毕。
        
    def read_sac_BHZ(self):  #读取路径下的所有Z分量的SAC数据,实例化类的时候直接运行这个,获得self.sac_info_all.setdefault字典，存储头文件信息
        sac_files = glob.glob(self.sac_file_path+'/*.BHZ.sac')
        for sac_file in sac_files:
            sac_name = os.path.basename(sac_file)
            st = read(sac_file)
            self.sac_info = Sac_info(st[0]) #读取sac头文件中的所有信息
            tem_list = [self.sac_info.dist,self.sac_info.mag] #临时列表，存储头文件的震中距，震级
            self.sac_info_all.setdefault(sac_name,[]).extend(tem_list)         
            
            if self.get_noise: #剪切数据，长度为60s
                process_noise = Noise_process(st[0],self.sac_file_path+'/noise_data/',sac_name)
            
    def build(self):
        self.read_sac_BHZ()   #直接调用这个子函数。
        print ('bulid is running')
    
    def plot_sac(self,file_path,save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        sac_files = glob.glob(file_path+'/*.BHZ.sac')
        for sac_data in sac_files:
            name = os.path.basename(sac_data)
            st = read(sac_data) 
            save_name = save_path+name.replace('BHZ.sac','png')  
            st.plot(equal_scale=False,outfile=save_name,color='red',size=(1800,1000)) 
          
    

#sac_data = Sac_files('/home/zhangzhipeng/software/github/2020/data',get_noise=True)


'''       
#遍历sac数据，统计头文件信息，形成字典，键是文件名，值是头文件信息(震中距，震级)            
sac_data = Sac_files('/home/zhangzhipeng/software/github/2020/data')
#print(sac_data.sac_info_all)

AI_out_name  = '/home/zhangzhipeng/software/github/2020/program/picks.csv' #存储Phasenet结果的csv文件
phasenet_out = {}  #字典，键是震中距处以50，值是2个数,一个是正确拾取(误差在0.5s以内的数量)，一个是错误拾取的数量

with open(AI_out_name) as f:
    reader = csv.reader(f)
    header = next(reader)
    for row in reader:
        name,tp = row[0],row[1]
        file_name = row[0].replace('npz','BHZ.sac')
        if file_name in sac_data.sac_info_all.keys(): 
            sac_header = sac_data.sac_info_all[file_name]  #对应文件名称的头文件信息
            dist = int(sac_header[0])//50                  #震中距(float32),处以50转换成，0，1，2，3，4，5
            if dist not in phasenet_out.keys():
                phasenet_out.setdefault(dist,[]).extend([0,0])            
            if row[1]=='[]':
                phasenet_out[dist][1]+=1
            else:
                pick_nums = re.findall('\d+',row[1]) #row[1]是str--'[1448]'，找到其中的数字，并转换成整数，有的有2个拾取
                pick_num = int(pick_nums[0])
                #print (dist,pick_num)
                if abs(pick_num-1500) <= 50:
                    phasenet_out[dist][0]+=1
                else:
                    phasenet_out[dist][1]+=1

print (phasenet_out)
'''

'''
phasenet_dict = {0: [5333, 211], 3: [830, 147], 1: [2730, 269], 2: [973, 308], 5: [104, 12], 4: [298, 58], 6: [71, 0], 8: [15, 0], 9: [10, 1], 7: [17, 0], 51: [3, 0], 15: [3, 0], 32: [1, 0], 43: [1, 0], 18: [1, 0], 12: [3, 0], 22: [1, 0], 47: [1, 0], 10: [4, 0], 35: [2, 0], 11: [5, 0], 40: [2, 0], 50: [3, 0], 34: [2, 0]}

fp_dict = {1: [2336, 663], 0: [4787, 757], 2: [942, 339], 3: [801, 176], 4: [282, 74], 5: [95, 21], 8: [13, 2], 6: [65, 6], 15: [2, 1], 7: [16, 1], 12: [2, 1], 11: [5, 0], 9: [9, 2], 35: [2, 0], 10: [3, 1], 40: [2, 0], 43: [1, 0], 50: [3, 0], 47: [1, 0], 18: [0, 1], 34: [2, 0], 32: [1, 0], 22: [1, 0], 51: [3, 0]}

import matplotlib.pyplot as plt

#画准确率随震中距的变化，输入一个字典，键是震中距处以50，值是2个数,一个是正确拾取(误差在0.5s以内的数量)，一个是错误拾取的数量
def dist_accuracy(ai_result,fp_result,png_name):
    tem_ai = []  #不同震中距准确率(0,1,2,3分别代表50，100，150，200及以上的震中距，对应的数值一个是正确拾取的数量，一个是错误拾取的数量)
    tem_fp = []
    for i in range(4):
        if i in ai_result.keys():
            tem_ai.append(ai_result[i])
            tem_fp.append(fp_result[i])
    
    for i in ai_result.keys():
        if i>3:
            value = ai_result[i]
            tem_ai[3][0]+=value[0]
            tem_ai[3][1]+=value[1]
    for i in fp_result.keys():
        if i>3:
            value = fp_result[i]
            tem_fp[3][0]+=value[0]
            tem_fp[3][1]+=value[1]
    
    print (tem_ai,tem_fp)
    
    ai_list = []
    fp_list = []
    for i in tem_ai:
        accuracy = round(i[0]/(i[0]+i[1]),2)
        ai_list.append(accuracy) #准确性减去一个数值，这样画图更好看些。
    for i in tem_fp:
        accuracy = round(i[0]/(i[0]+i[1]),2)
        fp_list.append(accuracy) #准确性减去一个数值，这样画图更好看些。
    
    print (ai_list,fp_list)
    #画图
    name_list = ['50','100','150','200 and above']  

    a =list(range(len(ai_list))) #x坐标  
    x = np.array(a)

    total_width, n = 0.8, 2  
    width = 0.1 
  
    plt.bar(x-width/2, ai_list, width=width, label='ai',fc = 'y')  

    plt.bar(x+width/2, fp_list, width=width, label='fp',fc = 'r')  

    x_label = [0,1,2,3] #在哪个位置写x轴标签
    y_label = [0.3,0.6,0.9]

    plt.xticks(x_label,name_list)  #写x轴标签,x_label坐标同时也是x轴的锚点，如果没有后面的name_list，则显示的x轴坐标是x_label的内容
    plt.yticks(y_label)


    plt.legend() 
    plt.savefig(png_name)
    plt.show()  
    plt.close()

dist_accuracy(phasenet_dict,fp_dict,'phasenet_accuracy_dist.png')
'''



'''
import json

#字典，存储FP结果，键是文件名，值是0或1，1代表拾取误差在0.5s以内，
filename1 = 'Plot_result.json'
with open(filename1) as file_obj:
    fp_result = json.load(file_obj)

sac_data = Sac_files('/home/zhangzhipeng/software/github/2020/data')
#print(sac_data.sac_info_all)

fp_out = {}  #字典，键是震中距处以50，值是2个数,一个是正确拾取(误差在0.5s以内的数量)，一个是错误拾取的数量

keys = list(fp_result.keys())
for sac_name in keys:
    if sac_name in sac_data.sac_info_all.keys(): 
        sac_header = sac_data.sac_info_all[sac_name]  #对应文件名称的头文件信息
        dist = int(sac_header[0])//50    
        if dist not in fp_out.keys():
            fp_out.setdefault(dist,[]).extend([0,0])            
        if fp_result[sac_name][0]==0:
            fp_out[dist][1]+=1
        elif fp_result[sac_name][0]==1:
            fp_out[dist][0]+=1
            
print (fp_out)
'''


class Dog():
    def __init__(self,name,age):
        self.name = name
        self.age  = age
        self.sex  = 'boy'  #属性不需要形参来定义，可以直接在这里赋值。
        
    def roll(self,update_age):
        #print ('test %s'%(self.age))
        self.age += update_age
        return self.age
        
my_dog  = Dog('black',12)
new_age = my_dog.roll(2)
print (new_age)

class New_dog(Dog):
    def __init__(self,name,age):  #调用父类的方法，让子类可以调用父类的属性
        #super().__init__(name,age)
        super(New_dog,self).__init__()
        self.num = 1
        
your_dog = New_dog('white',20)
print (your_dog.age)

'''
your_dog = Dog('white',22)
print (your_dog.sex)


class Leg():  #新建一个类，将其实例化后作为属性给字类New_dog
    def __init__(self):
        self.left_leg  = 12
        self.right_leg = 123
        
    def sum_leg(self):
        print(f"there are two leg sum is {self.left_leg+self.right_leg}")



class New_dog(Dog):
    def __init__(self,name,age):  #调用父类的方法，让子类可以调用父类的属性
        super().__init__(name,age) #特殊函数，可以调用父类的方法
        self.leg = Leg() 
        
    def roll(self):
        print ('new class do not need this method') #定义一个与父类相同的方法，会覆盖父类中的方法。
    
her_dog = New_dog('red',120)
print (her_dog.sex)

her_dog.roll()
print (her_dog.leg.left_leg)
her_dog.leg.sum_leg()

'''


















