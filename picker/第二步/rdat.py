#!/usr/bin/python
# -*- coding:UTF-8 -*-
from datetime import *
import os
'''
fa=open('manual_picksy.dat','r')
all1=fa.readlines()
for line1 in all1:
    arg=line1.split()
    print (arg[5],type(arg[5]))

'''

'''
time1='2019-06-17T15:01:30.26Z' 
form=datetime.strptime(time1,"%Y-%m-%dT%H:%M:%S.%fZ")
print (form)
name1='2019.168.14.52.51.0000'
name2=name1.split('.')
oyear=int(name2[0]) #起始时间 #
month=1
day=1    #起始时间是2019年1月1日，然后再加上天数(减去一)
addday=int(name2[1])-1
hour=int(name2[2])
minutes=int(name2[3])
second=int(name2[4])
otime=datetime(oyear,month,day,hour,minutes,second)+timedelta(addday)
print (otime) #这个就是最终的文件名的时间换算成的年月日时间。
subtime=form-otime  #两个时间相减
print (subtime.days)  #相减后的结果，注意没有hours和minutes属性。
print (subtime.seconds)
print (dir(subtime)) #查看subtime所拥有的属性，可以看到其没有hours和minutes属性。
#print (help(subtime.days))
print (hasattr(subtime,"hours"))#查看subtime是否有hours这个属性。
a=[1,2,3]
print (dir(a))
print (hasattr(a,"append")) #
#print (help(hasattr))
'''
'''
用法：直接运行就行，会读取本目录下的sac文件的文件名，并读取其时间信息，然后读取test文件(里面是fileterpicker生成的test)并将test中的时间减去sac文件名的时间
'''

path='/home/zhangzhipeng/software/filterpicker/picker/第二步' #注意修改路径
ls=os.listdir(path)
for name in ls:
    if len(name)>30:
        nt=name.split('.')
        #print(nt[8])
        if nt[8]=='BHZ': #如果文件名字的后缀是BHZ
            oyear,hour,addday,minutes,second=int(nt[0]),int(nt[2]),int(nt[1])-1,int(nt[3]),int(nt[4])
            #sta,net=nt[7],nt[6]
            otime=datetime(oyear,1,1,hour,minutes,second)+timedelta(addday)
            #print (otime,nt[7],nt[6]) #将后缀为BHZ的台站名称中的时间转换成需要格式，台站名称，台网名称。
            #读取自动挑选出的picktime，并且减掉文件名称中的起始时间
            fa=open('test','r')       
            all1=fa.readlines()
            for line1 in all1:
                arg=line1.split()
                time1=arg[6][0:4]+'-'+arg[6][4:6]+'-'+arg[6][6:8]+'T'+arg[7][0:2]+':'+arg[7][2:4]+':'+arg[8] #读取的pick时间转换成需要的格式
                ptime=datetime.strptime(time1,"%Y-%m-%dT%H:%M:%S.%f")
                subtime=ptime-otime
                print (time1,subtime,subtime.seconds)

            fa.close()
'''
                if arg[0]==sta and arg[1]==net : 
                    #print (arg[5],type(arg[5]))	
                    picktime1=datetime.strptime(arg[5],"%Y-%m-%dT%H:%M:%S.%fZ") #一共手动挑选了5个到时。
                    picktime2=datetime.strptime(arg[6],"%Y-%m-%dT%H:%M:%S.%fZ")
                    picktime3=datetime.strptime(arg[7],"%Y-%m-%dT%H:%M:%S.%fZ")
                    picktime4=datetime.strptime(arg[8],"%Y-%m-%dT%H:%M:%S.%fZ")
                    picktime5=datetime.strptime(arg[9],"%Y-%m-%dT%H:%M:%S.%fZ")
                    subtime1=picktime1-otime #挑选的5个到时减去起始时间就是走时。
                    subtime2=picktime2-otime
                    subtime3=picktime3-otime
                    subtime4=picktime4-otime
                    subtime5=picktime5-otime
                    print (subtime1.seconds,subtime2.seconds,subtime3.seconds,subtime4.seconds,subtime5.seconds) #挑选的第一个时间与文件名称中的时间的差。
                    '''




