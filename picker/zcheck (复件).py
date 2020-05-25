#!/usr/bin/python
# -*- coding:UTF-8 -*-

import subprocess
import numpy as np


time1=['0227','1455','1509','1521','1523','1536']
time2=['18','49','6','4','20']
filterwindow1=300
for longt in range(290,301):
    for tup1 in range(25,32):
        for t1 in np.arange(10,15,0.5):
            for t2 in np.arange(10,15,0.5):
                subprocess.call('cd /home/zhangzhipeng/software/filterpicker/picker;./picker_func_test mer1.sac zmer1  %s %s %s %s %s' %(filterwindow1,longt,tup1,t1,t2),shell=True)
                fa=open('zmer1')
                a1=fa.readlines()
                fa.close()
                i=0
                picknum=0
                for line1 in a1:
                    picknum+=1
                    part=line1.split()
                    if part[7] in time1:
                        second=part[8].split('.')
                        if second[0] in time2:
#                            fb=open('zresult.txt','a+') 这3行是写入匹配的时间
#                            fb.write(part[7]+' '+part[8]+'\n')
#                            fb.close()
                             #print (part[7],part[8])   
                             i+=1
                print (i)
                fb=open('zresult.txt','a+')
                fb.write(str(i)+' '+str(picknum)+' '+str(filterwindow1)+' '+str(longt)+' '+str(tup1)+' '+str(t1)+' '+str(t2)+'\n')
                fb.close()
                subprocess.call('cd /home/zhangzhipeng/software/filterpicker/picker;rm zmer1',shell=True)  

#存储zresult1中的结果,并排序
list1=[] 
list2=[]
f1=open('zresult.txt','r')
a2=f1.readlines()
f1.close()
for line1 in a2:
    p2=line1.split()
    list1.append(p2[0])
    list1.append(p2[1])
    list1.append(p2[2])
    list1.append(p2[3])
    list1.append(p2[4])
    list1.append(p2[5])
    list1.append(p2[6])
    list2.append(list1)
    list1=[]
list_sort=sorted(list2,key=lambda x:(x[0],x[1],x[2],x[3],x[4],x[5],x[6]))
for i in range(len(list2)):
    f2=open('znew.txt','a+')
    f2.write(list_sort[i][0]+' '+list_sort[i][1]+' '+list_sort[i][2]+' '+list_sort[i][3]+' '+list_sort[i][4]+' '+list_sort[i][5]+' '+list_sort[i][6]+'\n')
    f2.close()
subprocess.call('cd /home/zhangzhipeng/software/filterpicker/picker;rm zresult.txt',shell=True)  


#本程序在pick运行完一次后使用，会读取结果文件zmer1,找到与time1和time2中的时间，将其输出到zresult.txt中，然后会删除zmer1
'''
#HWS    -12345 BHZ  ? P1_    ? 20190617 0227   18.0792 GAU 2.000e-02 0.000e+00 2.411e+01 1.000e-02
#   0       1    2   3  4      5  6       7       8         
#建立存储标记到时的列表
time1=['0227','1455','1509','1521','1523','1536']
time2=['18','49','6','4','20']
fa=open('zmer1')
a1=fa.readlines()
fa.close()
i=0
for line1 in a1:
    part=line1.split()
    if part[7] in time1:
        second=part[8].split('.')
        if second[0] in time2:
            fb=open('zresult.txt','a+')
            fb.write(part[7]+' '+part[8]+'\n')
            fb.close()
            #print (part[7],part[8])   
            i+=1
print (i)
fb=open('zresult.txt','a+')
fb.write(str(i)+'\n')
fb.close()
subprocess.call('cd /home/zhangzhipeng/software/filterpicker/picker;rm zmer1',shell=True)
'''























