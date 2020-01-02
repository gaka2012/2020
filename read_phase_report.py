#!/usr/bin/python
# -*- coding:UTF-8 -*-

#1 以下程序会读取report_path路径下所有的震相报告，提取其中符合时间以及经纬度的相应台站的震中距和走时，存放到out_path中 
#  命名规则是台站-震相.dat
#2 读取out_path中与st对应的台站走时表，绘制走时表图，每个台站一张。存放到plot_path中。
# Pg:红  Pn:黑  Sg：蓝 Sn:绿
import re,os,glob,sys
from obspy.core import UTCDateTime
import matplotlib.pyplot as plt
import numpy as np

st=['WDT','ZHQ','JZG','QCH','PWU','JMG','ZJG','AXI','WCH','CD2','YZP','BAX','MDS','TQU','GZA','HMS']
#st          = ['HWS','ANZ']         #需要寻找的台站
phase       = ['Pg','Sg','Pn','Sn','P','S'] #需要寻找的震相
report_path = 'report-file-2018-2019'       #震相报告所在路径
out_path    = 'dist-time'                   #将结果文件移动到此路径下
lat_min     = 29
lat_max     = 34
lon_min     = 101
lon_max     = 107
begin_time  = UTCDateTime('2017-01-01 00:00:00')
end_time    = UTCDateTime('2019-12-31 00:00:00')
plot_travel_time=True                      #是否画图
plot_path   = 'picture'                    #画图之后移动到此位置。


earth_num=0 #震相报告中的地震数量
phase_num=0 #最终挑选的震相的数量
earth=False
report_files= glob.glob(report_path+'/*')
total=len(report_files)
num=0
for report_file in report_files:
   line_num=0
   num+=1
   percent=num/total
   sys.stdout.write("\r{0}{1}".format("#"*10 , '%.2f%%' % (percent * 100)))
   sys.stdout.flush()
   with open(report_file,'r',encoding='gbk') as fr:
       for line in fr:
           i=0
           line_num+=1
           part=line.split()
           try:
               if part[0]=='DBO' : #DBO表示是某个地震
                   bjtime  = UTCDateTime(part[2]+' '+part[3]) #发震时刻
                   if lat_min<=float(part[4])<=lat_max and  lon_min<=float(part[5])<=lon_max and  begin_time<=bjtime<=end_time:
                       earth=True #符合条件才会继续寻找相应的拾取台站到时信息
                       gmttime = bjtime-8*3600
                       earth_num+=1
                   else :
                       earth=False
               if earth:
                   if part[2] in st and (line[24:28].split()[0] in phase):
                       try:
                           arrival_time = UTCDateTime(line[34:56])
                           travle_time  = arrival_time-bjtime #走时
                           dist         = float(line[65:71])  #震中距
                           out_name     = part[2]+'-'+line[24:28].split()[0]+'.dat'
                           fa=open(out_name,'a+')
                           fa.write(str(dist)+' '+str(travle_time)+'\n')
                           fa.close()
                           phase_num+=1
                       except Exception as e: #可能会遇到没有震中距的情况，dist=float就会报错。所以用了except
                           #print (report_file,line_num) #可以显示报错所在的文件及行数。
                           continue
           except IndexError:
               continue
os.system('mv *.dat %s'%(out_path))    
print ('total earthquake ==',earth_num)
print ('total phase      ==',phase_num)


#画走时表
exist_phase=0 #统计存在多少震相
if plot_travel_time:
    for sta in st:
        #每个台站画一幅图,一幅图包括多个震相的散点图及拟合的曲线。
        max_dist   =0 #最大dist,用来作为x轴最大值
        max_travel =0 #最大走时，用来作为y轴最大值
        try:
            plt.figure(figsize=(25,15)) 
            for ph in phase:
                tem_dist   = []
                tem_travel = []
                #由于部分台站可能没有人工拾取，所以首先要判断是否存在。
                name        = sta+'-'+ph+'.dat'
                result_path = os.path.join(out_path,name)
                if os.path.exists(result_path):
                    exist_phase+=1
                    with open(result_path,'r') as fr:
                        for line in fr:
                            part=line.split()
                            tem_dist.append(float(part[0]))
                            tem_travel.append(float(part[1]))
                    #根据dist和time画图，标签是震相名称
                    if ph=='Pg':
                        color='r'
                    elif ph=='Pn':
                        color='k'
                    elif ph=='Sg':
                        color='b'
                    elif ph=='Sn':
                        color='g'
                    plt.scatter(tem_dist,tem_travel,s=10,c=color,label=ph)
                    tem_max_dist   = max(tem_dist)
                    tem_max_travel = max(tem_travel)
                    #print (max_dist)
                    if tem_max_dist>max_dist:
                        max_dist=tem_max_dist
                    if tem_max_travel>max_travel:
                        max_travel=tem_max_travel
                    tem_dist,tem_travel=[],[]            
                else:
                    pass
            #print (max_dist)
            plt.xticks(np.arange(0,max_dist+50,10))   
            plt.yticks(np.arange(0,max_travel+10,10)) 
            plt.legend()
            plt.xlabel('dist',fontsize=25)
            plt.ylabel('travel-time',fontsize=25)
            plt.title(sta,fontsize=25)
            plt.savefig(sta)
            plt.close()
        except Exception as e:
            print (e)
        #print (sta,len(dist),len(dist[0])) #每个台站都有2个列表，一个存储dist,一个存储time.并且是嵌套的，嵌套不同的震相。
    print ('exist_phase ==',exist_phase)
    os.system('mv *.png %s'%(plot_path))




















