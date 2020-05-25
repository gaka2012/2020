#!/usr/bin/python
# -*- coding:UTF-8 -*-

import time
import subprocess
import numpy as np
import os
import progressbar
import datetime
#注意修改路径1

#1读取所有生成的结果文件，找到有几个符合的，以及总的pick数目，参数情况。无序的，写入zread1.txt
#time1=['2019-06-17T02:27:17.7','2019-06-17T14:55:50.3','2019-06-17T15:09:06.3','2019-06-17T15:21:04.9','2019-06-17T15:23:36.8','2019-06-17T15:36:06.6']
time1=['2019-06-17T16:39:16.7','2019-06-17T18:58:23.2','2019-06-17T20:11:12.7','2019-06-17T20:04:34.6','2019-06-17T21:04:07.7','2019-06-17T21:49:19.8','2019-06-18T00:14:57.1','2019-06-18T00:40:00.5','2019-06-18T05:23:46.9','2019-06-18T06:38:40.4','2019-06-18T10:54:03.1']
path='/home/zhangzhipeng/software/filterpicker/picker/long00' #注意修改路径1
ls=os.listdir(path)
p=progressbar.ProgressBar()


for i in p(range(len(ls))):
    name=ls[i]        
    name1=path+'/'+name  
    fa=open(name1,'r')
    a1=fa.readlines()
    fa.close()
    if len(a1)<80:
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
                for m in time1:
                    manual1=datetime.datetime.strptime(m,"%Y-%m-%dT%H:%M:%S.%f")
                    manual=time.mktime(manual1.timetuple())
            #如果挑选的时间与手动时间相差为+-2秒，则。。。
                    if  -2<= (picktime2-manual)<=2:
                        i+=1
            except IndexError:
                print (name)
        longt,tup,t1,t2=name.split('-')      
        fb=open('zread1.txt','a+')
    #有几个符合的，以及总的pick数目，参数情况。
        fb.write(str(i)+' '+str(picknum)+' '+longt+' '+tup+' '+t1+' '+t2+'\n')
        fb.close()        
    #tup参数可能是1一位数。所以如果第4个数小于4,那么tup参数就是2位的，否则就是一位的。
        if int(name[3])<5:
            longt=name[3:5]     
    #t1参数可能是2位数，比如12,23,31等，所以如果第5个数小于4,那么t1参数就是2位的，否则就是一位的。
            if int(name[5])<4:
                t1=name[5:7]
                t2=name[7:]
            else:
                t1=name[5]
                t2=name[6:]
    #如果第4个数大于4,说明longt是一位数。
        elif int(name[3])>=5: 
            longt=name[3]
            if int(name[4])<4:
                t1=name[4:6]
                t2=name[6:]
            else:
                t1=name[4]
                t2=name[5:]     
        
        fb=open('zread1.txt','a+')
        #有几个符合的，以及总的pick数目，参数情况。
        fb.write(str(i)+' '+str(picknum)+' '+name[0:3]+' '+longt+' '+t1+' '+t2+'\n')
        fb.close()
    else:
        continue
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
subprocess.call('cd /home/zhangzhipeng/software/filterpicker/picker;rm zread1.txt',shell=True)   




###########################################################################################
#3 读取第二步产生的有序的结果，按照实际符合清空数目分开，存储在00中。
fa=open('znew.txt')   #注意修改1
a3=fa.readlines()
fa.close()
for line1 in a3:
    pt=line1.split()
    for i in range(10):
        if int(pt[0])==i:
            path1='/home/zhangzhipeng/software/filterpicker/picker/00/' #注意修改2
            path2=path1+str(i)
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









    
    
    
    
