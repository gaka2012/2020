#!/usr/bin/python
# -*- coding:UTF-8 -*-

import time
import subprocess
import numpy as np
import os,glob
import progressbar
import datetime
#注意修改路径1

#1读取所有生成的结果文件，找到有几个符合的，以及总的pick数目，参数情况。无序的，写入zread1.txt



pick_file = 'mer2.txt' #手动挑选出的地震到时
path1     = '/home/zhangzhipeng/software/filterpicker/picker' #存放结果的路径
file1     = 'mer2test' #这个是run_filterpicker.py中生成的，需要读取的存放结果的文件夹
out_file  = 'mer2result'
max_pick  =  500  #每个文件中的pick的数量如果大于等于这个值则不读取。


#根据手动挑选出的地震到时的文件生成标准的地震到时列表
time_list=[] 
with open(pick_file,'r') as f:
    for line in f:
        part=line.split()
        time_list.append(part[3])


#第一步，得到结果路径中所有的文件名，存放在name_list中，并输出其长度
path =path1+'/'+file1+'/*' 
name_list=[]
refiles=glob.glob(path)
for path in refiles:
    name=os.path.basename(path)
    long1=name.split('-')
    name_list.append(path)
print (len(name_list))



for i in range(len(name_list)):
    name1=name_list[i]        
    namefile=os.path.basename(name1)
    #print (name1)
    fa=open(name1,'r')
    a1=fa.readlines()
    fa.close()
    if len(a1)<max_pick:
        i=0
        picknum=0
        for line1 in a1:
            part=line1.split()
            picknum+=1
        #跑程序得到的时间结果，转化为秒数，然后减去手动挑选的时间秒数。
            try :
                picktime=part[6]+part[7]+part[8]
                picktime1=datetime.datetime.strptime(picktime,"%Y%m%d%H%M%S.%f")
                picktime2=time.mktime(picktime1.timetuple())
        #手动拾取的时间转化为秒数。
                for m in time_list:
                    manual1=datetime.datetime.strptime(m,"%Y-%m-%dT%H:%M:%S.%fZ")
                    manual=time.mktime(manual1.timetuple())
            #如果挑选的时间与手动时间相差为+-2秒，则。。。
                    if  -2<= (picktime2-manual)<=2:
                        i+=1
            except Exception as e:
                fb=open('error.txt','a+')
                fb.write(name+' '+str(e)+'\n')
                fb.close()
        longt,tup,t1,t2=namefile.split('-')      
        fb=open('zread1.txt','a+')
    #有几个符合的，以及总的pick数目，参数情况。
        fb.write(str(i)+' '+str(picknum)+' '+longt+' '+tup+' '+t1+' '+t2+'\n')
        fb.close()        
    else:
        pass

#2存储zread1.txt中的结果,并排序,写入znew.txt,并删除zread1.txt
list1=[] 
list2=[]
f1=open('zread1.txt','r')
a2=f1.readlines()
f1.close()
for line1 in a2:
    p2=line1.split()
    list1.append(int(p2[0]))
    list1.append(int(p2[1]))
    list1.append(int(p2[2]))
    list1.append(int(p2[3]))
    list1.append(int(p2[4]))
    list1.append(int(p2[5]))
    list2.append(list1)
    list1=[]

list_sort=sorted(list2,key=lambda x:(x[0],x[1],x[2],x[3],x[4],x[5]))

#print (list_sort)
for i in range(len(list2)):
    f2=open('znew.txt','a+')
    f2.write(str(list_sort[i][0])+' '+str(list_sort[i][1])+' '+str(list_sort[i][2])+' '+str(list_sort[i][3])+' '+str(list_sort[i][4])+' '+str(list_sort[i][5])+'\n')
    f2.close()
#
subprocess.call('rm zread1.txt',shell=True)   



###########################################################################################
#3 读取第二步产生的有序的结果，按照实际符合清空数目分开，存储在00中。
out_path=os.path.join(path1,out_file)

if not os.path.exists(out_path):
    os.makedirs(out_path)

fa=open('znew.txt')   
a3=fa.readlines()
fa.close()
for line1 in a3:
    pt=line1.split()
    for i in range(len(time_list)+1):
        if int(pt[0])==i:
            path2=out_path+'/'+str(i)
            fb=open(path2,'a+')
            fb.write(line1)
            fb.close()









#以下程序会读取下载的正式观测报告report.txt，提取其中的地震目录，并将北京时间转换为gmt时间，提取HWS台站的手动到时，并将结果存放在manualreport.txt中。
# manualreport.txt中的数据格式如下：
#DBO SC 2019-06-24 10:11:30.83 2019-06-24 02:11:30.830000 ML 2.8 6 28.448 104.784 四川长宁
#             北京时间                  gmt时间               震级 深度 经纬度
#DPB SC   HWS BHZ          Pg 1  V 2019-06-24 05:36:21.52    1.75  102.2 218.6         
#           HWS台站手动拾取的到时
'''
import re
from datetime import *
fa=open('report.txt','r',encoding='gbk')
a1=fa.readlines()
fa.close()
i=0
for line1 in a1:
    part=line1.split()
    try:
        i+=1
        #寻找符合 2019-04-23 23:21:23.12 的
        time1=re.search('\d+\-\d+\-\d+',part[2])
        time2=re.search('\d+:\d+:\d+\.\d+',part[3])
        if time1 and time2:
            #识别北京时间，并将其转换为gmt时间
            bjtime=part[2]+'T'+part[3]
            readbj=datetime.strptime(bjtime,"%Y-%m-%dT%H:%M:%S.%f") 
            gmt1=readbj-timedelta(hours=8)
            #print (gmt1)
            fb=open('manualreport.txt','a+')
            fb.write(part[0]+' '+part[1]+' '+part[2]+' '+part[3]+' '+str(gmt1)+' '+part[7]+' '+part[8]+' '+part[6]+' '+part[4]+' '+part[5]+' '+part[16]+'\n')
            fb.close()
        #寻找手动拾取的HWS台站记录的到时。
        if part[2]=='HWS' and part[3]=='BHZ' and part[4]=='Pg':
            fb=open('manualreport.txt','a+')
            fb.write(line1)
            fb.close()
    except IndexError:
        i+=1
        #print (i)
        continue

'''

#subprocess.call('cd /home/zhangzhipeng/software/filterpicker/picker;rm zread1.txt',shell=True)   
#subprocess.call('shutdown -h 1',shell=True)









    
    
    
    
