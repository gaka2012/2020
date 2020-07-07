#!/usr/bin/python
# -*- coding:UTF-8 -*-

import glob,os
from obspy.core.stream import Stream
from obspy.core import read
from obspy.core import UTCDateTime



#根据挑出的图像(data_list)从未滤波的数据(sac_data)中来选择sac数据,并将信噪比改正为滤波后的信噪比并归一化到不确定性，将数据保存到(data)中。
data_path = '/home/zhangzhipeng/software/github/2020/program/sac_data' #未经滤波的sac数据的路径
save_path = '/home/zhangzhipeng/software/github/2020/program/data'     #将挑出来的数据保存到此路径下。
#1.根据图像名称得到sac文件名称，并按照图像文件名中的信噪比赋值。
#data_list = glob.glob("/home/zhangzhipeng/software/github/2020/program/data_figure/*.png") 
data_list = glob.glob("/home/zhangzhipeng/backup/data/pick_data/*.png")

for data in data_list:
    name     = os.path.basename(data) #SC.AXI_20180101062906_12_mag.png  
    s        = name.split('_')
    sac_name = "{}_{}.BHZ.sac".format(s[0],s[1]) #SC.AXI_20180101062906.BHZ.sac

    sac_path = os.path.join(data_path,sac_name)
    #snr      = float(s[2].replace('.png',''))
    snr      = float(s[2])
    
    print (snr)

    #将信噪比转换为不确定性。
    if snr<8:
        un = 0.5-(0.2/8*snr)
    elif  8<=snr<16:
        un = 0.3-(0.1/8*(snr-8)) 
    elif  16<=snr<24:
        un = 0.2-(0.1/8*(snr-16))
    elif snr>=24:
        un = 0.1
    un = round(un,2)
    
    #将数据复制到新的路径下，并修改t9,将不确定性写入t9
    os.system('cp %s %s'%(sac_path,save_path))
    os.chdir(save_path)
    st = Stream().clear()
    st+= read(sac_name)
    st[0].stats.sac.t9 = un
    st[0].write(sac_name,format='SAC')
    
    
'''
#读取(data)中的数据路径以及p波到时写入txt文件中。
data_path = '/home/zhangzhipeng/software/github/2020/program/data'
sac_list  = glob.glob(data_path+'/*.sac')
os.chdir(data_path)
for sac in sac_list:
    sac_name = os.path.basename(sac)  #SC.AXI_20180101062906.BHZ.sac
    st = Stream().clear()
    st+= read(sac_name)
    s  = st[0].stats.sac
    r_time = UTCDateTime(year=s.nzyear,julday=s.nzjday,hour=s.nzhour,minute=s.nzmin,second=s.nzsec,microsecond=s.nzmsec*1000) #参考时刻
    p_time = r_time+s.a #p波到时
    fa = open('test.txt','a+')
    fa.write(sac+' '+str(p_time)+'\n')
    fa.close()
'''
    
'''
 #读取(no_data)中的噪声数据路径以及-1234写入txt文件中。
data_path = '/home/zhangzhipeng/software/github/2020/program/no_data'
sac_list  = glob.glob(data_path+'/*.sac')

for sac in sac_list:
    fa = open('test.txt','a+')
    fa.write(sac+' '+'-1234'+'\n')
    fa.close()   
    
    
'''

  
    
    
    
    
    
    
    
    
    
    
    
    
    
    
