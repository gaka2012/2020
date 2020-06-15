#!/usr/bin/python
# -*- coding:UTF-8 -*-


import os,sys,glob,re,csv
from obspy.core import UTCDateTime
import csf_phase
from obspy.core import read
import matplotlib.pyplot as plt
from datetime import *   
import matplotlib.dates as mdate
import numpy as np
from itertools import groupby
from obspy.core import stream


'''
本程序会读取震相报告(输入4个参数)，生成基础字典，输出字典内容，画震级分布图。
根据提取的p波到时信息截取数据，存入gmt发震时间，到时，存成sac格式，文件名称是北京时间的发震时间。
'''
#计算信噪比，输入trace,P波到时，噪声取值长度，p波取值长度，返回计算后的信噪比
def cal_SNR(tr,tp,before_tp,after_tp): 
    #print (tr)
    #计算信噪比
    c_st = tr.copy()
    n_st = tr.copy()
    #1 剪切p波前后3秒，并转换为numpy格式数据
    no_data = n_st.trim(tp-before_tp,tp,pad=True, fill_value=0)
    no_data = np.asarray(no_data)
    p_data  = c_st.trim(tp, tp + after_tp,pad=True, fill_value=0)   #默认pad是false,这里是True，当给定的截取时间超出了数据范围会自动补fill_value,如果默认则不画超出的时间。
    p_data  = np.asarray(p_data)
    #2 numpy去绝对值
    ab_data = np.fabs(p_data) #P波取绝对值
    ab_nois = np.fabs(no_data)#噪声取绝对值
    #3 求p波最大值处以噪声均值
    p_max  = max(ab_data) #p波最大值
    n_mean = np.sum(ab_nois)/(len(ab_nois)) #噪声均值
    if n_mean==0:
        snr = -1234
    else:
        snr = round(p_max/n_mean,2)
    return snr
    
def cut_trace(tr,gt,cut_pro,cut_end,e_time,e_lat,e_lon,e_dep,e_mag,e_dis,e_azi,net,sta,S_phase,S_time,save_data): #输入一个trace,进行截取，并修改头文件信息
    c_tr = tr.copy()
    data = c_tr.trim(gt-cut_pro, gt+cut_end,pad=True, fill_value=0) 
    
    #修改头文件信息,发震时刻改为参考时刻，添加地震维度、经度、深度、震级、震中距、方位角、P到时、S到时。
    s = data.stats.sac
    ot = UTCDateTime(e_time)-8*3600 #发震时间转为gmt时间
    s.nzyear,s.nzjday,s.nzhour,s.nzmin,s.nzsec,s.nzmsec = ot.year,ot.julday,ot.hour,ot.minute,ot.second,int(ot.microsecond/1000) #发震时设为参考    
    s.evla,s.evlo,s.evdp,s.mag,s.dist,s.az = e_lat,e_lon,e_dep,e_mag,e_dis,e_azi
    s.a = gt-ot

    #计算信噪比
    snr = cal_SNR(tr,gt,10,4)
    s.t9 = snr
        
    #print (S_phase,type(S_phase))
    if S_phase=='-1234':
        s.t0 = -1234
        print ('PASS')
    else:
        s.t0 = S_time-8*3600-ot
    
    #print (s.t0)
    #保存成sac数据，文件名台网-台站-发震时刻, SC.AXI_20190101120001.sac
    ndate,ntime = e_time.split('T') 
    nyear,nmonth,nday = ndate.split('-') #
    nhour,nmin,nsec   = ntime.split(':')
    nsec = nsec.split('.')[0]
    nout =''.join([nyear,nmonth,nday,nhour,nmin,nsec])#20190101120001
    data_name = net+'.'+sta+'_'+nout+'.'+tr.stats.channel+'_noise.sac'  #如果是噪声则用这个命名
    data.write(data_name,format='SAC')
    #将数据移动到位置
    os.system('mv *.sac %s'%(save_data))

def plot_sac(sac_name): #输入一个截取好的sac文件名，(90s的单个文件)，画图并画到时和发震时刻竖线,生成的文件名是北京时间发震时刻去掉后面加上信噪比加上png
    print (sac_name)
    st = read(sac_name)
    tr=st[0] #原始数据
    #获取发震和p波到时
    s = tr.stats.sac
    ref_time = UTCDateTime(year=s.nzyear,julday=s.nzjday,hour=s.nzhour,minute=s.nzmin,second=s.nzsec,microsecond=s.nzmsec*1000)#参考时刻，即发震时刻。
    a = s.a #到时
    
    p_num = ref_time+a #
    p_num = datetime.strptime(str(p_num),"%Y-%m-%dT%H:%M:%S.%fZ") #到时绝对时刻
    o_num = datetime.strptime(str(ref_time),"%Y-%m-%dT%H:%M:%S.%fZ") #发震绝对时刻
    
    #print (p_num,type(p_num))

    #获取最大最下值，用来画竖线
    co=tr.copy()
    data_max=co.data.max()
    data_min=co.data.min()
    plt.figure(figsize=(25,15)) #设置生成的图片的默认尺寸，如果生成的图片太小，部分内容会被压缩，注意要在plt.subplot之前写这句话
    dt=tr.stats.starttime
    ax=plt.subplot(1,1,1)#行数，列数，第几个图
    #t=np.linspace(tr.stats.starttime-dt,tr.stats.endtime-dt,tr.stats.npts)#这里必须是减去dt，相当数x轴是从0开始，因为它不识别UTC格式的时间，需要其他方法转换

    #将坐标转换成时间，起点是其绝对时刻。
    t_list = []
    origin=datetime.strptime(str(dt), "%Y-%m-%dT%H:%M:%S.%fZ")
    for i in range(tr.stats.npts):
        add_second=0.01
        newtime=origin+timedelta(seconds=add_second*i)
        t_list.append(newtime)
    #print(t_list)
    #只显示特定的坐标，包括发震、p波到时
    #s波到时
    t0= s.t0 #s波到时
    if t0 == -1234:
        show_list = [t_list[0],o_num,p_num,t_list[9000]]
        #print(t0)
    else:
        t0    = s.t0
        s_num = ref_time+t0
        s_num = datetime.strptime(str(s_num),"%Y-%m-%dT%H:%M:%S.%fZ") #s波到时绝对时刻
        show_list = [t_list[0],o_num,p_num,s_num,t_list[9000]]
        #print(ref_time+8*3600,s_num)
        #print (show_list)
        plt.vlines(s_num,data_min,data_max,colors='r')

    #2条竖线
    plt.vlines(o_num,data_min,data_max,colors='g') 
    plt.vlines(p_num,data_min,data_max,colors='r') #画竖线，  x,    y_min ,  y_max , 颜色
    #标题
    #print(s.mag,type(s.mag),s.t9,type(s.t9),s.dist,type(s.dist))
    sac_name = sac_name.split('/')[-1] #如果文件名包含路径，则删除路径
    t_name = '.'.join(sac_name.split('.')[0:2]) #SC.AXI_20180101225925
    t_name = t_name+'.'+tr.stats.channel+'-'+str(s.mag)+'-'+str(s.dist)+'-'+str(s.t9)
    ax.set_title(t_name,color='red',fontsize=24) #添加小标题
    #坐标格式
    ax.set_xticks(show_list) 
    ax.set_xticklabels(show_list,rotation=20,fontsize=15)
    ax.xaxis.set_major_formatter(mdate.DateFormatter('%Y/%m/%d %H:%M:%S'))
    plt.plot(t_list,tr.data)
    plt_name = '.'.join(sac_name.split('.')[0:2]) #SC.AXI_20180101225925
    plt_name+= '_'+str(s.t9)+'.png'
    plt.savefig(plt_name)
    #plt.show()        #展示，可以2选1
    plt.close()       #关闭，否则会占内存

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
    

data_path = '/home/zhangzhipeng/datatest'  #存放原始数据的位置，下一级目录是SC
#data_path = '/bashi_fs/centos_data/sac_sc'

#

dict_test  = True  #输出字典内容
mag_distri = False  #画震级分布图
cut        = False  #截取数据,并计算信噪比，要自己输入p波和噪声的时间长度，计算完后存放在t9中。
p_s_diff   = False   #震中距分布图。

save_data = '/home/zhangzhipeng/software/github/2020/program/test' #截取后的数据保存位置

cut_pro,cut_end   = 120,0     #以P波为基准，提前30s，延后60s进行截取
phase_list   = ['Pg','Pn','P','Pb'] #截取P波震相
s_phase_list = ['Sg','S','Sn']  #截取S波震相

#首先输入数据调用class类，共4个数据。寻到符合台网-台站以及震相列表的地震手动拾取。
report_path  = '/home/zhangzhipeng/report-temporary' #存放震相报告的路径
file_list    = ['2018-01.txt']
#file_list    = ['2018-01.txt','2018-02.txt','2018-03.txt','2018-04.txt','2018-05.txt','2018-06.txt','2018-07.txt','2018-08.txt','2018-09.txt','2018-10.txt','2018-11.txt','2018-12.txt',] 
#sta_list  = ['AXI']
sta_list = ['AXI']
net_list = ['SC']


if __name__=="__main__":

    if not os.path.exists(save_data):
        os.makedirs(save_data)    
    #1. 读取震相报告文件，使用基础类csf.返回基础属性，一个字典。该字典只有符合sta_list和net_list的地震信息。
    csf1 = csf_phase.Csf(report_path,file_list,net_list,sta_list) #输入的参数是震相报告的路径，震相报告文件列表，台网列表，台站列表。

    #输出csf的初始变量，一个字典 phase_dict;里面是一个地震事件对应的多个地震到时。都是class类。
    #print (csf1.phase_dict) #这个输出的是一个字典，里面都是class,看不到具体信息。
    sta_pha_time = {}
    for key,value in csf1.phase_dict.items():  #csf1.phase_dict是一个字典，键值是每一个地震事件，对应的数据是所有的震相到时。    
         #输出键值(DBO类)-输出对应的数据(DPB类)[列表]
            #print (key.eq_data,key.eq_ot,key.eq_lat,key.eq_lon,key.eq_dep,key.eq_mag,ke.mag_type)   
        stored_phase = [] # 可能同一个台站即拾取了Pg,又拾取了Pn，为了防止这种现象，设一个临时列表。
        for i in range(len(value)):    #循环每一个地震事件，如果地震事件拾取的台网、台站、震相符合列表，则存储。    
            if value[i].Phase_name in phase_list:
                net,sta   = value[i].Net_code,value[i].Sta_code
                net_sta   = value[i].Net_code+'-'+value[i].Sta_code
                phase     = value[i].Phase_name
                mag       = key.mag_value
                dist      = value[i].Distance            
                if net_sta in stored_phase:
                    pass
                else:
                    stored_phase.append(net_sta)        
                    #tem_list存储地震事件信息(发震时间、纬度、经度、深度、震级、震中距、方位角、震级类型)
                    if len(key.eq_ot)==0:
                        tem_list = [-1234]
                    else:
                        tem_list = [key.eq_ot]
                    eq_list  = [key.epi_lat,key.epi_lon,key.epi_dep,key.mag_value,value[i].Distance,value[i].azi] #防止是空
                    for num in eq_list:
                        try:
                            tem_list.append(float(num))
                        except ValueError:
                            tem_list.append(-1234)
                    if len(key.mag_type)==0:
                        tem_list.append(-1234)
                    else:
                        tem_list.append(key.mag_type)
                        
                    #tem_list添加p波震相以及到时
                    tem_list.append(phase)#因为有前面的if,这个肯定没有问题
                    try:
                        pick_time = UTCDateTime(value[i].Phase_date+' '+value[i].Phase_time)
                        tem_list.append(pick_time)
                    except Exception:
                        tem_list.append(-1234)
                        
                    #找到P波到时后，循环某个地震事件的所有拾取，找到同一个台站的S波到时,由于只需要找到一个就行，可以break
                    for j in range(len(value)): 
                        if value[j].Net_code==net and value[j].Sta_code==sta and value[j].Phase_name in s_phase_list:
                            #print ('test')\
                            #添加S波震相以及到时
                            s_phase     = value[j].Phase_name
                            tem_list.append(s_phase)
                            #前面已经添加震相了，这里主要是添加s到时信息，如果出现错误，即s到时拾取错误，则添加-1234
                            try:
                                s_pick_time = UTCDateTime(value[j].Phase_date+' '+value[j].Phase_time)
                                tem_list.append(s_pick_time)
                            except Exception:
                                tem_list.append(-1234)
                            break
                    #补充字典，键值是台网-台站，值是tem_list
                    sta_pha_time.setdefault(net_sta,[]).append(tem_list)
                
    #测试，输出字典中的内容
    if dict_test:
        for key,value in sta_pha_time.items(): #输出一个字典，键值是台网-台站，
                                       #对应的值是发震时间(str)、纬度(float)、经度、深度、震级、震中距、方位角、震级类型
                                       #P波震相和到时(UTC)以及S波震相和到时，北京时间，要减去8个小时。
                
            print (key)
            for i in value:
                print (i)
            #print (len(value))
    
    #画震级分布图
    if mag_distri:
        for key,value in sta_pha_time.items():
            print (key)  #台网-台站
            mag_list = []
            for event in value: #value是一个列表套列表，里面是每一个地震事件信息以及该台站的拾取信息
                mag = event[4] 
                if mag!=-1234:
                    mag_list.append(mag)    
            plot_num_distribution(mag_list,key)
    
    #画震中距分布图：
    if p_s_diff:
        for key,value in sta_pha_time.items():
            p_s_list = []
            for event in value:
                try:
                    p_dis = event[5]
                    if p_dis!=-1234:
                        p_s_list.append(p_dis)
                except IndexError:
                    pass
            #print (sorted(p_s_list))
            plot_num_distribution(p_s_list,key,10,10,'snr','num')

    if cut:
        for key,values in sta_pha_time.items():
            fa=open('process.txt','a+') #打开这个文件查看开始跑的台站
            fa.write(key+'\n')
            fa.close()   
            num_r = 0 #正确截取的地震事件数量
            num_l = 0 
            try:
                ns       = key.split('-') #将SC-AXI分开
                net,sta  = ns[0],ns[1]
                sta_path = os.path.join(data_path,ns[0],ns[1]) #/home/zhangzhipeng/SC/AXI
            
                 #key是台网、台站。
                #value是每一个地震事件的拾取。
                #然后循环每一个到时,找到到时所对应的文件，比如时间2018-05-04，对应的文件是2018/05/04
                #截取的数据时间是gmt时间,但是最后存储的时候文件名是bj时间，实际数据是没有时间的，只有点数。
                for ct in values:
                    #print (ct[0])
                    #分配发震时间、纬度、经度、深度、震级、震中距、震级类型(str,float)、P波震相、到时(str,UTCDateTime)
                    try:
                        e_time,e_lat,e_lon,e_dep,e_mag,e_dis,e_azi,e_type,P_phase,P_time,S_phase,S_time= ct[0],ct[1],ct[2],ct[3],ct[4],ct[5],ct[6],ct[7],ct[8],ct[9],ct[10],ct[11] #
                    except IndexError:
                        e_time,e_lat,e_lon,e_dep,e_mag,e_dis,e_azi,e_type,P_phase,P_time = ct[0],ct[1],ct[2],ct[3],ct[4],ct[5],ct[6],ct[7],ct[8],ct[9]
                        S_phase,S_time = '-1234',-1234
                    #print (e_time,P_phase,type(P_phase),P_time,type(P_time),S_phase,type(S_phase))
            
                    #按照P波pick_time北京时间寻找文件名进行截取，S波到时只是用来画图
                    bj = P_time
                    year,month,day = '%02d'%(bj.year),'%02d'%(bj.month),'%02d'%(bj.day) #2018,5,4
                    gdata = os.path.join(sta_path,year,month,day) #路径：/home/zhangzhipeng/datatest/SC/AXI/2018/05/04
            
                    #按照pick_time的gmt时间寻找数据
                    gt = bj-8*3600
            
                #判断数据是否存在，由于原始数据只有18、19年的部分数据，所以可能存在没有数据的情况。
                    if not os.path.exists(gdata):
                        print ('no such file',gdata)
                    elif len(os.listdir(gdata))!=3:
                        print ('sac channel is wrong',gdata)
                    else:
                #读取三分量数据,并按照ENZ来排序,
                        st_3c = glob.glob(gdata+'/*.SAC')
                        st_3c = sorted(st_3c,key=lambda i :(re.search(r'BH.',i).group(0)))
                        st = read(st_3c[0])
                        st+= read(st_3c[1])
                        st+= read(st_3c[2])  
                        
                        #读取三分量中实际数据的通道名称，看是否是3分量，有时可能是三个BHE分量。
                        c_name = [tr.stats.channel for tr in st] #['BHE', 'BHZ', 'BHN'] 
                        c_name = '_'.join(c_name)
                        if 'BHE' not in c_name or 'BHN' not in c_name or  'BHZ' not in c_name:
                            print('sac channel is missed', gdata)
                        
                        else:
                            #如果截取的时间在单个文件所在天数24小时范围内，直接截取  
                            s_time   = bj-24*3600 #
                            nor_time = UTCDateTime(s_time.year,s_time.month,s_time.day,16,00,00)#将时间归一化到16:00:00
                            nor_end  = nor_time+24*3600
                            if  (gt-cut_pro)>=nor_time and (gt+cut_end)<=nor_end:
                                #对3分量文件分别进行截取，防止某个通道的时间不够，需要merge
                                #print (gdata)
                                for tr in st:
                                    num_r+=1
                                    cut_trace(tr,gt,cut_pro,cut_end,e_time,e_lat,e_lon,e_dep,e_mag,e_dis,e_azi,net,sta,S_phase,S_time,save_data)
                                    
                            #截取的时间 stat-30在，end不在
                            elif (gt-cut_pro)>=nor_time and (gt+cut_end)>nor_end:
                                try:
                                #print ('wrong time',gdata)
                                    num_r+=1
                                    bj = bj+24*3600 #跳到第二天
                                    year,month,day = '%02d'%(bj.year),'%02d'%(bj.month),'%02d'%(bj.day) #2018,5,4
                                    gdata = os.path.join(sta_path,year,month,day) #路径：/home/zhangzhipeng/datatest/SC/AXI/2018/05/04
                                    #读取三分量数据,并按照ENZ来排序,与之前读取的st_3c合并
                                    new_st_3c = glob.glob(gdata+'/*.SAC')
                                    new_st_3c = sorted(new_st_3c,key=lambda i :(re.search(r'BH.',i).group(0)))
                                    tem_st = read(new_st_3c[0])
                                    tem_st+= read(new_st_3c[1])
                                    tem_st+= read(new_st_3c[2])
                                    #3分量与前一天的数据合并
                                    for i in range(3):
                                        st_c = stream.Stream()
                                        st_c.append(st[i])
                                        st_c.append(tem_st[i])
                                        st_c.sort(['starttime'])
                                        st_c.merge(method=1,interpolation_samples=0,fill_value=0)
                                        cut_trace(st_c[0],gt,cut_pro,cut_end,e_time,e_lat,e_lon,e_dep,e_mag,e_dis,e_azi,net,sta,S_phase,S_time,save_data)
                                except IndexError:
                                    num_r-=1
                                    print ('no such data',gdata)
                                    for tr in st:
                                        num_r+=1
                                        cut_trace(tr,gt,cut_pro,cut_end,e_time,e_lat,e_lon,e_dep,e_mag,e_dis,e_azi,net,sta,S_phase,S_time,save_data)
                            #截取的时间 start-30不在，end在
                            elif  (gt-cut_pro)<nor_time and (gt+cut_end)<=nor_end:
                                try:
                                    #print ('wrong time',gdata)
                                    num_r+=1
                                    bj = bj-24*3600
                                    year,month,day = '%02d'%(bj.year),'%02d'%(bj.month),'%02d'%(bj.day) #2018,5,4
                                    gdata = os.path.join(sta_path,year,month,day) #路径：/home/zhangzhipeng/datatest/SC/AXI/2018/05/04
                                    #读取三分量数据,并按照ENZ来排序,与之前读取的st_3c合并
                                    new_st_3c = glob.glob(gdata+'/*.SAC')
                                    new_st_3c = sorted(new_st_3c,key=lambda i :(re.search(r'BH.',i).group(0)))
                                    tem_st = read(new_st_3c[0])
                                    tem_st+= read(new_st_3c[1])
                                    tem_st+= read(new_st_3c[2])
                                    #3分量与前一天的数据合并
                                    for i in range(3):
                                        st_c = stream.Stream()
                                        st_c.append(st[i])
                                        st_c.append(tem_st[i])
                                        st_c.sort(['starttime'])
                                        st_c.merge(method=1,interpolation_samples=0,fill_value=0)
                                        #save_data1 = '/home/zhangzhipeng/software/github/2020/program/sac_data1'
                                        cut_trace(st_c[0],gt,cut_pro,cut_end,e_time,e_lat,e_lon,e_dep,e_mag,e_dis,e_azi,net,sta,S_phase,S_time,save_data)
                                #找前一天的数据可能没有此数据，会出现IndexError错误,此时强行补零截取
                                except IndexError:
                                    num_r-=1
                                    print ('no such data',gdata)
                                    for tr in st:
                                        num_r+=1
                                        cut_trace(tr,gt,cut_pro,cut_end,e_time,e_lat,e_lon,e_dep,e_mag,e_dis,e_azi,net,sta,S_phase,S_time,save_data)
                                
            except Exception as e : #所有异常，输出到文件中
                print ('test')
                fb=open('err.txt','a+')
                fb.write(str(e)+'\n')
                fb.write(key+' '+e_time+'\n')
                fb.close()

        print ('cut %s data'%(num_r))
















































