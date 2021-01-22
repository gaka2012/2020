#!/usr/bin/python
# -*- coding:UTF-8 -*-

import os,glob,json,csv,re
from obspy.core import read
from datetime import *
from obspy.core import UTCDateTime
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import entropy as sci_entropy


#phasenet 5.1
#将cut_data中截取后的长度为120s的sac数据(3个分量)转换为npz数据，shape是3000，3，转存到npz_data中。并生成相应的csv文件
'''
data_path  = '/home/zhangzhipeng/software/data' #截取过的长度为3001个点的sac三分量文件 
npz_path   = '/home/zhangzhipeng/software/github/2020/PhaseNet-master/dataset/one_pred'   #将sac三分量转存成npz数据后保存位置。

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
    
    
os.chdir('/home/zhangzhipeng/software/github/2020/PhaseNet-master/dataset')
f = open('one.csv','w',encoding='utf-8')
csv_writer = csv.writer(f)
csv_writer.writerow (['fname','itp','its'])  #写入表头
for key,value in save_dict.items():
    #print (key,value[0][0])
    csv_writer.writerow([key,value[0][0],value[0][1]]) #写入实际数据
f.close()   

print('convert %s sac data to npz dat'%(data_num))
'''


    
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
#phasenet 6.1.1

'''
#将李君截取的数据命名方式修改成与phasnet一致，数据格式一致，标签一致；看笔记,将数据移动到phasenet下作训练
npz_datas = glob.glob('/home/zhangzhipeng/data/SC_picked/*.npz')
save_path = '/home/zhangzhipeng/software/github/2020/PhaseNet-master/dataset/SC_train/'
for npz in npz_datas:
    #修改文件名
    file_name = os.path.basename(npz)
    new_name = save_path+file_name.replace('_EV','').replace('.','_',1)
    A = np.load(npz)        
    data,itp,its,cha = A['data'].T,A['itp'],A['its'],'BHE_BHN_BHZ'

    np.savez(new_name,data=data,itp=itp,its=its,channels=cha)

#根据得到的数据作csv文件
f = open('SC_train.csv','w',encoding='utf-8')
csv_writer = csv.writer(f)
csv_writer.writerow (['fname','itp','its','channels'])  #写入表头
npz_datas = glob.glob('/home/zhangzhipeng/software/github/2020/PhaseNet-master/dataset/SC_train/*.npz')
for npz in npz_datas:
    #获得npz数据的文件名，将后缀.npz去掉，作为参数传递给画图子函数。
    file_name = os.path.basename(npz)
    file_name = file_name.replace('_EV','')
    A = np.load(npz)
    itp,its = A['itp'],A['its']
    csv_writer.writerow([file_name,itp,its,'BHE_BHN_BHZ']) #写入实际数据
f.close() 
'''

#标签 phasenet 6.3.1
#将一天的sac文件转存成mseed文件，并补零    
'''
#将四川台网一个台站一个月的数据每天生成一个npz文件，由于有些数据的起始时间不是16：00，并且三分量的数据不齐，使用trim补零，生成的是npz，8640001，3的数据
data_path  = '/home/zhangzhipeng/data/SC/AXI/2018/01' #遍历一个月的所有数据 
save_path   = '/home/zhangzhipeng/data/mseed_data/'   #将sac三分量转存成mseed数据后保存位置。

data_files = sorted(glob.glob(data_path+'/*'))       #一个月的所有天数
for day in data_files:
    #读取每一天下的所有sac数据,设置切割的起始时间是16:00:00
    st = read(day+'/*.SAC') 
    st.sort(keys=['channel'], reverse=False) #对三分量数据按照时间排序
    t   = st[0].stats.starttime 
    dt = UTCDateTime(t.year,t.month,t.day,16) 
    
    #复制并切割数据，切割后的数据是3分量的8640001个点，转存成mseed格式
    c_st = st.copy()
    data = c_st.trim(dt, dt + 24*3600,pad=True, fill_value=0) 
    print (data)
    name = save_path+'_'.join(day.split('/')[-5:])+'.mseed'
    data.write(name)


#统计文件夹下每一个mseed文件名称，并写入csv文件中

name_list = []
fnames = glob.glob('/home/zhangzhipeng/data/mseed_data/*.mseed')
for name in fnames:
    fname = os.path.basename(name)
    name_list.append(fname)

f = open('mseed_continue.csv','w',encoding='utf-8')
csv_writer = csv.writer(f)
csv_writer.writerow (['fname','E','N','Z'])  #写入表头
for i in name_list:
    csv_writer.writerow([i,'BHE','BHN','BHZ']) #写入实际数据
f.close() 
'''


#标签 phasenet 6.3.2 
#data_reader.py中的子函数 def read_mseed(self,fp,channels)测试。
'''
fp = 'SC_AXI_2018_01_01.mseed'
meta = read(fp)
meta = meta.detrend('constant')
meta = meta.merge(fill_value=0)
meta = meta.trim(min([st.stats.starttime for st in meta]), 
                 max([st.stats.endtime for st in meta]), 
                 pad=True, fill_value=0)
nt = len(meta[0].data)
channels = ['BHE','BHN','BHZ']
data = [[] for ch in channels]

for i, ch in enumerate(channels):
    tmp = meta.select(channel=ch)
    #print (tmp) 
    if len(tmp) == 1:
        data[i] = tmp[0].data
    elif len(tmp) == 0:
        print(f"Warning: Missing channel \"{ch}\" in {meta}")
        data[i] = np.zeros(nt)
    else:
        print(f"Error in {tmp}")
data = np.vstack(data)
print(data.shape)          

pad_width = int((np.ceil((data.shape[1] - 1) / 3000))*3000 - data.shape[1])
print (pad_width)
if pad_width == -1:
      data = data[:,:-1]
else:
      data = np.pad(data, ((0,0), (0, pad_width)), 'constant', constant_values=(0,0))
print (data.shape)

data = np.hstack([data, np.zeros_like(data[:,:3000//2]), data[:,:-3000//2]])
print (data.shape)
data = np.reshape(data, (3, -1, 3000))
data = data.transpose(1,2,0)[:,:,np.newaxis,:]
print (data.shape)
'''

#标签 phasenet 6.3.3
#读取phasenet生成的结果-csv文件

'''
import csv

i=0
file_name = '/home/zhangzhipeng/software/github/2020/PhaseNet-master/output/picks.csv'
with open(file_name) as f:
    reader = csv.reader(f)  #读取csv文件内容
    header = next(reader)   #返回文件的下一行，类型是列表
    #print(header)
   
    #遍历前1000行，pass掉没有P拾取的行数，只寻找开头是SC_AXI_2018_01_02.mseed的数据
    for row in reader:
        if re.search(r'SC_AXI_2018_01_02.mseed',row[0]):
            if row[1]=='[]':
                pass
            else:
                print (row)
'''
    
'''
        i+=1
        if i >1000:
            break
        else:
            if row[1]=='[]':
                pass
            else:
                if re.search(r'SC_AXI_2018_01_02.mseed',row[0]):
                    num = row[0].split('_')[-1]
                    print (row)
                    print (num,type(row[1]))  

'''




#标签 phasenet 6.3.4对比自动拾取的结果和人工拾取的结果。
#1.1 读取out中生成的csv文件，并获取其有数据！！的行
#ob是文件名称的组合成或，进行正则匹配

'''
out_name  = '/home/zhangzhipeng/software/github/2020/PhaseNet-master/output/picks.csv' #存储结果的csv文件
data_name = '/home/zhangzhipeng/software/github/2020/PhaseNet-master/demo/mseed_continue.csv' 
names = []

#1.1.1先获取原始数据中的所有文件名称
with open(data_name) as f:
    reader = csv.reader(f)
    header = next(reader)
    for row in reader:
        names.append(row[0])

#1.1.2打开存储结果的csv文件，遍历，找到根原始数据文件名匹配的那一行，pass掉没有P拾取的行数，找到拾取的点数与文件名后缀相加，得到最终的点数。
#得到的字典键值是结果文件中的文件名，值是到时点数(里面有很多点数，因为拾取了很多的点)
ob = '|'.join(names)
out_dict = {}
with open(out_name) as f:
    reader = csv.reader(f)  #读取csv文件内容
    header = next(reader)   #返回文件的下一行，类型是列表
    #print(header)
   
    #遍历前1000行，，只寻找开头是SC_AXI_2018_01_02.mseed的数据
    for row in reader:
        out_result = re.search(ob,row[0]) #out_result[0]是搜索得到的结果
        if out_result:
            if row[1]=='[]':
                pass
            else:
                pick_nums = re.findall('\d+',row[1]) #row[1]是str--'[1448]'，找到其中的数字，并转换成整数，有的有2个拾取
                
                last_num = int(row[0].split('_')[-1])
                
                for point in pick_nums:
                    if last_num<8640000:
                        pick_num = int(point)+last_num
                    else:
                        pick_num = int(point)-8640000+last_num-1500
                        out_dict.setdefault(out_result[0],[]).append(pick_num)
          
          
#1.1.3根据字典中的数据对读取原始数据画图，先全部画完，集中到save_bad_picture文件夹中。
save_good_picture = '/home/zhangzhipeng/data/figure/phasenet/good/'
save_bad_picture  = '/home/zhangzhipeng/data/figure/phasenet/bad/'
data_path         = '/home/zhangzhipeng/software/github/2020/PhaseNet-master/demo/mseed_test/'      
for key,value in out_dict.items():
    print (key,value[0])
    #读取原始数据，画图,起始时间是文件的起始时间加上点数/100,起始时间前30s和后30s
    st = read(data_path+key)
    st.sort(keys=['channel'], reverse=False) #对三分量数据排序
    co=st.copy()
    #去均值，线性，波形歼灭,然后滤波
    co.detrend('demean').detrend('linear').taper(max_percentage=0.05, max_length=10.)
    co=co.filter('bandpass',freqmin=1,freqmax=15) #带通滤波
    for i in value:
        png_name = save_bad_picture+key+'_'+str(i)+'.png'
        h0 = co[0].stats.starttime+i/100
        co.plot(starttime= (h0-30) ,endtime=(h0 + 30),equal_scale=False,outfile=png_name,color='red',size=(1800,1000))
        
#1.1.4读取震相报告获得人工拾取
#加载字典,里面是AXI台站一个月2018/01的震相报告，不管发震时刻，只把p波到时提取出来，一个月才94个地震。
filename='info_dict.json'
with open(filename) as file_obj:
    p_s_dict = json.load(file_obj)
   
p_list = [] 
for key,value in p_s_dict.items():
    for cons in value:
         p_list.append(cons[9])
    
#.1.1.5将p波到时中的年月日作为字典的键值，完整到时转换为UTC格式作为value
#.2读取某天的所有value,即p波到时，计算与起始时间的差值(每天的起始时间是00:00:00.00),得到的是与起始时间的点数(取整，1s一个点)。   
p_dict = {}
for p in p_list:
    dates = p.split('T')[0]
    p_dict.setdefault(dates,[]).append(UTCDateTime(p))
    #print (dates)
       

#得到一个新的字典，键值是年月日，value是p波到时的点数
people_dict = {}
for key,value in p_dict.items():
    start_time = UTCDateTime(key) #将每天的2018-01-31T00:00:00.000000Z作为起始时间(实际上可能差个0.01s,感觉没啥影响)
    p_num = []
    for p_arr in value:
            delta = int((p_arr-start_time)*100)-1
            p_num.append(delta)
            #$print(delta)            
    people_dict.setdefault(key,[]).append(p_num)
    
#得到最终的比对结果字典，键值是日期，值是有几个对上的(good_num)，几个没有对上的,人工拾取有几个
#将所有的自动拾取进行画图，对上的放在一个文件夹中，没有对上的放在另一个文件夹中。

result_dict = {}
for key,value in people_dict.items():
    name = 'SC_AXI_'+'_'.join(key.split('-'))+'.mseed'  #与人工拾取的字典中的key值与自动拾取的字典中的key值一致
    good_num = 0
    phase_nums = 0
    #如果人工拾取的键值(日期)与自动拾取的键值(日期)一致，则循环人工拾取，
    if name in out_dict:  
        phase_pick = out_dict[name] #自动拾取的结果，是一系列点数
        #遍历人工拾取
        for human_pick in value[0]:
            good_out = [i for i in phase_pick if abs(i-human_pick)<200]    
            #如果拾取的误差在2s以内
            if good_out: 
                good_num += 1
                print (key,human_pick,good_out)
                
                #如果对上了，则将图片移动位置
                old_png_name = save_bad_picture+name+'_'+str(good_out[0])+'.png'
                new_path     = save_good_picture
                os.system('mv %s %s'%(old_png_name,new_path))
        phase_nums = len(phase_pick) #自动拾取的总数
        png_name = save_bad_picture+key+'_'+str(i)+'.png'
    else:
        print (f"this file is missed {name}")
    if good_num:
        tem = [good_num,phase_nums-good_num,len(value[0])]
        #print (good_num,phase_nums)    
        result_dict.setdefault(name,[]).append(tem)
print (result_dict)

'''


'''
#最终的比对结果字典，键值是日期，值是有几个对上的(good_num)，几个没有对上的,人工拾取有几个
#进行画图，2个柱状图，左边是人工拾取的数量，右边是自动拾取的数量
def plot_double_bar(result_dict):
    name_list = [] #x轴名称，可以不用
    data_1,data_2,data_3 = [],[],[]
    width = 0.2
    
    for key,value in result_dict.items():
        name_list.append('.'.join(key.split('.')[0].split('_')[2:]))
        data_1.append(value[0][0])
        data_2.append(value[0][1])
        data_3.append(value[0][2])
    x = list(range(len(data_3)))
    plt.bar(x, data_3, label='boy',color='green',width=0.2) 
    
    for i in range(len(x)):
        x[i] = x[i]+width 
    plt.bar(x, data_1, label='boy',color='blue',width=0.2)  
    plt.bar(x, data_2, bottom=data_1, label='girl',tick_label = name_list,color='red',width=0.2)  
    plt.savefig('test') 
    plt.close()
    
plot_double_bar(result_dict)    
    
'''



#标签 test1
#将90s的sac文件转存成mseed文件
'''
#将四川台网一个台站一个月的数据每天生成一个npz文件，由于有些数据的起始时间不是16：00，并且三分量的数据不齐，使用trim补零，生成的是npz，8640001，3的数据
data_path  = '/home/zhangzhipeng/software/sac_data' #遍历一个月的所有数据 
save_path   = '/home/zhangzhipeng/software/sac_data/mseed_data/'   #将sac三分量转存成mseed数据后保存位置。

data_files = sorted(glob.glob(data_path+'/*.BHZ.sac'))       #一个月的所有天数
for day in data_files:
    #读取每一天下的所有sac数据,设置切割的起始时间是16:00:00
    st = read(day.replace('BHZ.sac','*')) 
    st.sort(keys=['channel'], reverse=False) #对三分量数据按照时间排序

    start = st[0].stats.starttime    
    c_tr = st.copy()
    data = c_tr.trim(start+90, start+180,pad=True, fill_value=0) 

    name = save_path+os.path.basename(day).replace('BHZ.sac','mseed')
    
    data.write(name)


#统计文件夹下每一个mseed文件名称，并写入csv文件中

name_list = []
fnames = glob.glob('/home/zhangzhipeng/software/sac_data/mseed_data/*.mseed')
for name in fnames:
    fname = os.path.basename(name)
    name_list.append(fname)

f = open('mseed_continue.csv','w',encoding='utf-8')
csv_writer = csv.writer(f)
csv_writer.writerow (['fname','E','N','Z'])  #写入表头
for i in name_list:
    csv_writer.writerow([i,'BHE','BHN','BHZ']) #写入实际数据
f.close() 

'''

#标签test2.1 根据字典中的文件名生成csv文件。
'''
filename='/home/zhangzhipeng/software/github/2020/90_s_event_info.json'
with open(filename) as file_obj:
    info = json.load(file_obj)
keys = list(info.keys())


f = open('one.csv','w',encoding='utf-8')
csv_writer = csv.writer(f)
csv_writer.writerow (['fname','E','N','Z'])  #写入表头

for key in keys:
    csv_writer.writerow([key.replace('BHZ.sac','mseed'),'BHE','BHN','BHZ']) #写入实际数据
f.close()   
'''
#标签test2.2 ai识别的结果，并将识别错误的，需要所有的地震事件名称，csv结果文件。

'''
filename='./mseed_data/90_s_event_info.json'
out_name = './output/picks.csv'
data_path = '/bashi_fs/centos_data/zzp/120_data' #原始数据位置
no_pick    = '/bashi_fs/centos_data/zzp/120_data/cp_no_pick'
wrong_pick = '/bashi_fs/centos_data/zzp/120_data/cp_wrong_pick'

with open(filename) as file_obj:
    info = json.load(file_obj)

keys = list(info.keys())   #'SC.PWU_20180329020838.BHZ.sac

print('there are %s datas '%(len(keys)))


#读取phasenet跑的连续波形数据(90s)的结果，建立字典，键是mseed文件名称，值是相对应的拾取点数的第一个，
#然后将结果保存下来，以备画图。



out_dict = {}
with open(out_name) as f:
    reader = csv.reader(f)  #读取csv文件内容
    header = next(reader)   #返回文件的下一行，类型是列表
    #print(header)
   
    for row in reader:
        name = row[0] #'SC.AXI_20180104103927.mseed_0'
        name_point = name.split('_')[-1] #文件名称的后缀 0,3000,6000等
        file_name  = name.replace('_'+name_point,'') #文件名 SC.AXI_20180104103927
        
        if row[1]=='[]':
                pass
        else:
            #找到AI拾取的结果
            pick_nums = re.findall('\d+',row[1]) #返回一个列表，里面是str格式的拾取，有的有2个拾取
             
            if int(name_point)<9000:
                pick_num = int(name_point)+int(pick_nums[0])
            else:
                pick_num  = int(pick_nums[0])+int(name_point)-10500
            
            #更新字典，键是文件名，值是最小的到时拾取
            if file_name not in out_dict.keys():                
                out_dict[file_name] = pick_num
            else:
                if abs(out_dict[file_name]-3000) > abs(pick_num-3000):
                    out_dict[file_name] = pick_num
                    
                    
#print (out_dict)
pick_num = 0 #最后到的地震数量的总数
less_one = 0.3 #只有残差小于这个数的才会被装到列表中。
no_pick_num,less_one_num = 0,0
ai_result = [] #列表，里面是很多个元组，每个元组都是残差以及数值1(用来画图)
ai_dict,wrong_dict = {},{}
ai_keys = list(out_dict.keys())

#遍历所有的地震事件，如果ai拾取到了，pick_num加1，
for key in keys:
    key_name = key #SC.PWU_20180329020838.BHZ.sac
    key = key.replace('BHZ.sac','mseed')
    if key in ai_keys:
        pick_num+=1
        resudal = (out_dict[key]-3000)/100 #AI拾取到时与手动拾取的残差
        if abs(resudal)<=less_one:
            tem_list = [resudal,1]
            ai_result.append(tem_list) #误差元组，用来画图
            less_one_num +=1
            ai_dict[key] = resudal     #字典，值是文件名称，对应的值是误差，这个字典只有误差小于less_one的数据才会写入。
        else:
            wrong_dict[key_name] = resudal  #拾取误差大于0.5s的拾取，形成字典，键是文件名称，值是对应的拾取时间。

    else:
        no_pick_num+=1

print ('there are %s picks and %s resudal less than %s second'%(pick_num,less_one_num,less_one))
print ('%s data no picks'%(no_pick_num))

wrong_keys = list(wrong_dict.keys())
print ('there are %s wrong_picks '%(len(wrong_keys)))

#保存字典
filename='ai_result.json'
with open(filename,'w') as file_obj:
    json.dump(ai_result,file_obj)

'''


#1.5 sac单分量数据画图， 将数据格式转换为npz,进行滤波等处理，shape是1,9001，输入画完图的保存路径，文件名称，数据，tp到时

'''
def plot_waveform_npz(plot_dir,file_name,data,itp): 
    plt.figure(figsize=(25,15))
    data = data
    t=np.linspace(0,data.shape[1]-1,data.shape[1]) #(0,9000,9001)
    plt.plot(t,data[0,:])        
    data_max=data.max()
    data_min=data.min()
    tp_num = itp
    plt.vlines(tp_num[0],data_min,data_max,colors='blue') 
    plt.vlines(tp_num[1],data_min,data_max,colors='r') 
    
    title = str(tp_num[1])
    plt.suptitle(title,fontsize=25)
    
    png_name=plot_dir+'/'+file_name+'png' #保留的文件名是信噪比加后面的信息
    plt.savefig(png_name)
    plt.close()  


data_path  = '/home/zhangzhipeng/software/github/2020/data/wrong_pick'  
save_dir = '/home/zhangzhipeng/software/github/2020/data/wrong_pick/figure'  

data_files = sorted(glob.glob(data_path+'/*BHZ.sac'))       #一个月的所有天数
total = len(data_files)
num =0
for data in data_files:
    #遍历所有的z分量数据，并以此找到三分量数据
    file_name   = os.path.basename(data)
    figure_name = file_name.replace('BHZ.sac','')
    if file_name in wrong_keys:
    
        st = read(data)     
        co=st.copy()
        #去均值，线性，波形歼灭,然后滤波
        co.detrend('demean').detrend('linear').taper(max_percentage=0.05, max_length=10.)
        co=co.filter('bandpass',freqmin=1,freqmax=15) #带通滤波
        
        #将滤波后的数据转换成numpy格式，
        data=np.asarray(co)
        tp_ai   = int(wrong_dict[file_name]*100)
        tp_list = [3000,3000+tp_ai]
        print (tp_list)
                 
        plot_waveform_npz(save_dir,figure_name,data,tp_list)

'''





#将noise数据的文件名制成csv文件，准备跑phasenet

'''
data_path = '/home/zhangzhipeng/software/github/2020/PhaseNet-master/mseed_data/11419_data'
name_list = []
mseed_name = glob.glob(data_path+'/*.mseed')
for name in mseed_name:
    mseed = os.path.basename(name)
    name_list.append(mseed)

f = open('noise.csv','w',encoding='utf-8')
csv_writer = csv.writer(f)
csv_writer.writerow (['fname','E','N','Z'])  #写入表头

for key in name_list:
    csv_writer.writerow([key,'BHE','BHN','BHZ']) #写入实际数据
f.close()   
'''


#标签test2.2 ai识别噪声的结果，并将识别错误的，。

'''
out_name = './output/picks.csv'



#读取phasenet跑的连续波形数据(90s)的结果，建立字典，键是mseed文件名称，值是相对应的拾取点数的第一个，
#然后将结果保存下来，以备画图。

out_dict = {}
with open(out_name) as f:
    reader = csv.reader(f)  #读取csv文件内容
    header = next(reader)   #返回文件的下一行，类型是列表
    #print(header)
   
    for row in reader:
        name = row[0] #'SC.AXI_20180104103927.mseed_0'
        name_point = name.split('_')[-1] #文件名称的后缀 0,3000,6000等
        file_name  = name.replace('_'+name_point,'') #文件名 SC.AXI_20180104103927.mseed
        
        if row[1]=='[]':
                pass
        else:
            #找到AI拾取的结果
            pick_nums = re.findall('\d+',row[1]) #返回一个列表，里面是str格式的拾取，有的有2个拾取
             
            if int(name_point)<9000:
                pick_num = int(name_point)+int(pick_nums[0])
            else:
                pick_num  = int(pick_nums[0])+int(name_point)-10500
            
            #更新字典，键是文件名，值是最小的到时拾取
            if file_name not in out_dict.keys():                
                out_dict[file_name] = pick_num
            else:
                if abs(out_dict[file_name]-3000) > abs(pick_num-3000):
                    out_dict[file_name] = pick_num
                    

keys = list(out_dict.keys())
print('ai picks %s wrong noise'%(len(keys)))



sac_path = '/home/zhangzhipeng/software/github/2020/PhaseNet-master/mseed_data/11419_data' #BHZ.sac文件所在目录
save_path = '/home/zhangzhipeng/software/data/noise_data/ai_pick' #将ai错误拾取的noise复制到这个路径中
sacs = glob.glob(sac_path+'/*.mseed')
sacs = [os.path.basename(i) for i in sacs] #sac文件的所有名称

num = 0
os.chdir(sac_path)
#ai从noise中错误拾取的地震名称
for key in keys:
    sac_name = key
    if sac_name in sacs:
        num+=1
        os.system('cp %s %s'%(sac_name,save_path))


print('there are %s wrong picks'%(num))


#保存字典
filename='ai_noise.json'
with open(filename,'w') as file_obj:
    json.dump(out_dict,file_obj)

'''


'''
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
    data = tr
    
    event = data[num-preceding_time*fm:num-preceding_time*fm+duration_time*fm]  #15秒长的数据    
    event_part   = np.reshape(event,[-1,100])       #reshape一下，将其转换为n(15)个长度为100的片段。
    event_energy = energy_per_frame(event_part)  #得到15个片段的能量值
    event_out    = sci_entropy(event_energy) #求熵值,即15个平均能力的熵值。
    return round(event_out,2)

#画npz数据， 数据格式是npz,shape是3,9001，输入画完图的保存路径，文件名称，数据
def plot_waveform_npz(plot_dir,file_name,data,itp,entropy): 
    plt.figure(figsize=(25,15))
    ax=plt.subplot(data.shape[0],1,1) #(3,1,1) 输入的数据shape是3,9001
    print(data.shape)
    for j in range(data.shape[0]):
        plt.subplot(data.shape[0],1,j+1,sharex=ax)
        t=np.linspace(0,data.shape[1]-1,data.shape[1]) #(0,2999,3000)
        data_max=data[2,:].max()
        data_min=data[2,:].min()
        plt.plot(t,data[j,:])
        plt.vlines(itp,data_min,data_max,colors='r') 
        
    plt.suptitle(str(entropy),fontsize=25,color='r')
    png_name=file_name+'.png' 
    plt.savefig(plot_dir+png_name)
    #os.system('mv *.png png') 
    plt.close()  
    


filename='ai_noise.json'
with open(filename) as file_obj:
    ai_noise = json.load(file_obj)


mseed_path = '/home/zhangzhipeng/software/data/noise_data/ai_pick' #将ai错误拾取的noise复制到这个路径中,2065个
mseeds = glob.glob(mseed_path+'/*.mseed')

event = []
num = 0
for mseed in mseeds:
    mseed_name = os.path.basename(mseed)
    if mseed_name in ai_noise.keys():
        tp = ai_noise[mseed_name]
        st = read(mseed)    
        num+=1
        st.sort(keys=['channel'], reverse=False) #对三分量数据排序
    
        co=st.copy()
        #去均值，线性，波形歼灭,然后滤波
        co.detrend('demean').detrend('linear').taper(max_percentage=0.05, max_length=10.)
        co=co.filter('bandpass',freqmin=1,freqmax=15) #带通滤波
            
        #将滤波后的数据转换成numpy格式，
        data=np.asarray(co)    #(3,9001)   
        z_channel = data[2,:]  #(9001)
        
        try:
            entropy = calculate_entropy(z_channel,tp)
            event.append(entropy)
    
            plot_waveform_npz(mseed_path+'/figure/',mseed_name,data,tp,entropy)
        except ValueError:
            print(mseed_name)
print('there are %s event'%(num))

'''   
    
    
    
#读取phasenet跑的连续波形数据(90s)的结果，建立字典，键是mseed文件名称，值是相对应的拾取点数最好的一个，

'''
out_name = './output/picks.csv'
out_dict = {}
with open(out_name) as f:
    reader = csv.reader(f)  #读取csv文件内容
    header = next(reader)   #返回文件的下一行，类型是列表
    #print(header)
   
    for row in reader:
        name = row[0] #'SC.AXI_20180104103927.mseed_0'
        name_point = name.split('_')[-1] #文件名称的后缀 0,3000,6000等
        file_name  = name.replace('_'+name_point,'') #文件名 SC.AXI_20180104103927.mseed
        
        if row[1]=='[]':
                pass
        else:
            #找到AI拾取的结果
            pick_nums = re.findall('\d+',row[1]) #返回一个列表，里面是str格式的拾取，有的有2个拾取
             
            if int(name_point)<9000:
                pick_num = int(name_point)+int(pick_nums[0])
            else:
                pick_num  = int(pick_nums[0])+int(name_point)-10500
            
            #更新字典，键是文件名，值是最小的到时拾取
            if file_name not in out_dict.keys():                
                out_dict[file_name] = pick_num-3000
            else:
                if abs(out_dict[file_name]) > abs(pick_num-3000):
                    out_dict[file_name] = pick_num-3000
                    
                    
with open('ai_name_pick.json','w') as ob:
    json.dump(out_dict,ob)     

'''

#读取phasenet跑的连续波形数据(90s)的结果，建立字典，键是mseed文件名称，值是相对应的拾取点数最好的一个，

'''
out_name = './output/picks.csv'
out_dict = {}
p_big,pro = 0,0.8 #p波概率大于0.6的有几个
with open(out_name) as f:
    reader = csv.reader(f)  #读取csv文件内容
    header = next(reader)   #返回文件的下一行，类型是列表
    #print(header)
   
    for row in reader:
        name = row[0] #'SC.AXI_20180104103927.mseed_0'
        name_point = name.split('_')[-1] #文件名称的后缀 0,3000,6000等
        file_name  = name.replace('_'+name_point,'') #文件名 SC.AXI_20180104103927.mseed
        
        if row[1]=='[]':
                pass
        else:
            #找到AI拾取的结果
            pick_nums = re.findall('\d+',row[1]) #返回一个列表，里面是str格式的拾取，有的有2个拾取
            p = re.findall('\d+\.\d+',row[2])   #p波的概率，返回一个列表，里面是str格式的拾取，有的有2个
            if float(p[0])>pro:
                #print (p[0])
                p_big+=1
                
                if int(name_point)<9000:
                    pick_num = int(name_point)+int(pick_nums[0])
                else:
                    pick_num  = int(pick_nums[0])+int(name_point)-10500
            
                #更新字典，键是文件名，值是最小的到时拾取
                if file_name not in out_dict.keys():                
                    out_dict[file_name] = pick_num
                else:
                    if abs(out_dict[file_name]) < abs(pick_num):
                        out_dict[file_name] = pick_num

print ('there are %s noises probality bigger than %s'%(p_big,pro))
                    
      
with open('noise.json','w') as ob:
    json.dump(out_dict,ob)     
'''



#画npz数据， 数据格式是npz,shape是3,9001，输入画完图的保存路径，文件名称，数据
'''
def plot_waveform_npz(plot_dir,file_name,data,itp): 
    plt.figure(figsize=(25,15))
    ax=plt.subplot(data.shape[0],1,1) #(3,1,1) 输入的数据shape是3,9001
    print(data.shape)
    for j in range(data.shape[0]):
        plt.subplot(data.shape[0],1,j+1,sharex=ax)
        t=np.linspace(0,data.shape[1]-1,data.shape[1]) #(0,9000,9001)
        data_max=data[j,:].max()
        data_min=data[j,:].min()
        plt.plot(t,data[j,:])
        plt.vlines(itp,data_min,data_max,colors='r') 
        
    #plt.suptitle(str(entropy),fontsize=25,color='r')
    png_name=file_name+'.png' 
    plt.savefig(plot_dir+png_name)
    #os.system('mv *.png png') 
    plt.close()  
'''
#1.5 sac单分量数据画图， 将数据格式转换为npz,进行滤波等处理，shape是1,9001，输入画完图的保存路径，文件名称，数据，tp到时


def plot_waveform_npz(plot_dir,file_name,data,itp,num): 
    plt.figure(figsize=(25,15))
    data = data
    t=np.linspace(0,data.shape[0]-1,data.shape[0]) #(0,9000,9001)
    plt.plot(t,data)        
    data_max=data.max()
    data_min=data.min()
    tp_num = itp
    plt.vlines(tp_num,data_min,data_max,colors='red') 
    #plt.vlines(tp_num[1],data_min,data_max,colors='r') 
    
    #title = str(tp_num[1])
    plt.suptitle(str(num),fontsize=25)
    
    png_name=plot_dir+str(num)+'_'+file_name+'.png' #保留的文件名是信噪比加后面的信息
    #print (png_name)
    plt.savefig(png_name)
    plt.close()  

data_path = '/home/zhangzhipeng/software/github/2020/PhaseNet-master/mseed_data/11419_data'  #存放原始噪声数据的位置
with open('noise.json') as ob:
    noise = json.load(ob)

os.chdir(data_path)
keys = list(noise.keys())
num=0
for name in keys[:200]:
    st = read(name)    
    num+=1
    st.sort(keys=['channel'], reverse=False) #对三分量数据排序
    co=st.copy()
    #去均值，线性，波形歼灭,然后滤波
    co.detrend('demean').detrend('linear').taper(max_percentage=0.05, max_length=10.)
    co=co.filter('bandpass',freqmin=1,freqmax=30) #带通滤波
    #co=co.filter('highpass',freq=20)        
        #将滤波后的数据转换成numpy格式，
    data=np.asarray(co)    #(3,9001)  
    tp = noise[name]
    z_channel = data[2,:]  #(9001)
    #print (tp)
    part = z_channel[tp-2000:tp+2000]
    
    try:
        plot_waveform_npz('/home/zhangzhipeng/software/github/2020/PhaseNet-master/mseed_data/figure/',name,part,2000,num)
    except ValueError:
        print(name)
    #try:
    #    plot_waveform_npz('/home/zhangzhipeng/software/github/2020/PhaseNet-master/mseed_data/figure/',name,z_channel,tp)
    #except ValueError:
    #    print (name)
    
    #print (data.shape,type(noise[name]))














