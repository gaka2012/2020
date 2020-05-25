#!/usr/bin/python
# -*- coding:UTF-8 -*-

import time
import subprocess
import numpy as np
import threading
from multiprocessing import Pool
#修改1.带入反演的数据名称 mer1.sac
#修改2.生成的结果存放的位置，result2

def mulfilter(item):
    (name2,filterwindow1,longt,tup1,t1,t2)=item
    subprocess.call('cd /home/zhangzhipeng/software/filterpicker/picker;./picker_func_test mer2.sac /home/zhangzhipeng/software/filterpicker/picker/test00/%s  %s %s %s %s %s' %(name2,filterwindow1,longt,tup1,t1,t2),shell=True)
                            #要修改存储结果的文件就改这里
parameter=[]#传递给多线程的参数列表
list1=[] #存放6个参数的临时列表(第一个参数是文件名)
start=time.time()
filterwindow1=300
#4个参数的取值范围
for longt in range(120,130,10):  #1 280-301 15-32 6-20 6-20
    for tup1 in range(10,11): #300 300 15 10 15 #300 300 5-15 #注意不要小于5
        for t1 in np.arange(5,20,1):
            for t2 in np.arange(5,100,1):
                #根据4个参数命名结果文件
                name1=str(longt)+'-'+str(tup1)+'-'+str(t1)+'-'+str(t2)
                #name2=name1.replace('.','')
                #将6个参数打包，用于传递给多线程
                list1.append(name1)
                list1.append(filterwindow1)
                list1.append(longt)
                list1.append(tup1)
                list1.append(t1)
                list1.append(t2)
                parameter.append(tuple(list1))
                list1=[]
            

#print (parameter)

#多线程池
pool=Pool(5)
rl=pool.map(mulfilter,parameter)
pool.close()
pool.join()
end=time.time()
inter=end-start
print (inter)
#subprocess.call('shutdown -h 3',shell=True)




'''
                #调用程序
                subprocess.call('cd /home/zhangzhipeng/software/filterpicker/picker;./picker_func_test mer1.sac /home/zhangzhipeng/software/filterpicker/picker/result/%s  %s %s %s %s %s' %(name2,filterwindow1,longt,tup1,t1,t2),shell=True)

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

'''










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























