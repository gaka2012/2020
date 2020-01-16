#!/usr/bin/python
# -*- coding:UTF-8 -*-

import time,sys
import subprocess
import numpy as np
import os,glob
import progressbar
import datetime
from tqdm import tqdm,trange
import threading
from multiprocessing import Pool,Value,Process

#实现多进程的子函数
def mulread(item):
    name1 = item      
    namefile=os.path.basename(name1)
    answer_list = time_list.copy() #标准答案列表
    i=0
    picknum=0
    with open(name1,'r') as fr:
        for line1 in fr:
            part=line1.split()
            picknum+=1

        #跑程序得到的时间结果，转化为秒数，然后减去手动挑选的时间秒数。
            try :
                picktime=part[6]+part[7]+part[8]
                picktime1=datetime.datetime.strptime(picktime,"%Y%m%d%H%M%S.%f")
                picktime2=time.mktime(picktime1.timetuple())
        #手动拾取的时间转化为秒数。
                for m in answer_list:
                    manual1=datetime.datetime.strptime(m,"%Y-%m-%dT%H:%M:%S.%fZ")
                    manual=time.mktime(manual1.timetuple())
            #如果挑选的时间与手动时间相差为+-2秒，则。。。
                    if  -2<= (picktime2-manual)<=2:
                        i+=1
                        answer_list.remove(m)
                        break
            except Exception as e:
                fb=open('error.txt','a+')
                fb.write(name+' '+str(e)+'\n')
                fb.close()
          
        longt,tup,t1,t2=namefile.split('-')   
        result_list =[i,picknum,longt,tup,t1,t2]
        return result_list


a=time.time()
pick_file = 'mer3.txt' #手动挑选出的地震到时

datapath     = '/home/zhangzhipeng/software/github/2020/data' #存放数据的路径
resultpath   = '/home/zhangzhipeng/software/github/2020/result' #存放结果的路径 
#path1     = '/mnt/bashi_fs/filterpicker/picker' #存放结果的路径
file1        = 'mer3_600' #这个是run_filterpicker.py中生成的，需要读取的存放结果的文件夹
out_file     = 'mer3_600out' #最后生成的结果存放的文件。
max_pick     =  800  #每个文件中的pick的数量如果大于等于这个值则不读取。
min_pick     = 0     #每个文件中的pick的数量低于这个值也不读取。在下面的time_list中重新定义。
pool_num     = 8     #进程的数量


result_list  = []    #这个是生成的结果以列表的形式存在。

#根据手动挑选出的地震到时的文件生成标准的地震到时列表
time_list=[] 
with open(pick_file,'r') as f:
    for line in f:
        part=line.split()
        time_list.append(part[3])
min_pick  = int(len(time_list)/2)
time_list = sorted(time_list) 


###########################################################################################
#第一步，得到结果路径中所有的文件名，存放在name_list中，并输出其长度(文件的数量)
path =datapath+'/'+file1+'/*' 
name_list=[]
refiles=glob.glob(path)
for path in refiles:
    file_size = (os.path.getsize(path))/1000
    if  min_pick/10  < file_size < max_pick/10: #比如最大pick是800个，则文件大小大约是80kb，超过的说明pick大于800个了。
        name_list.append(path)
print (len(name_list))

total_file  = len(name_list)

mul_list=[]                          #将多个进程的结果收集起来                         
pool=Pool(pool_num)
#rl=pool.map(mulread,name_list)
iter = pool.imap(mulread,name_list)   #变成可以迭代的多线程
with tqdm (total=total_file ) as t:   #进度条
    for ret in iter:
        mul_list.append(ret)         #将每次的结果收集起来
        t.update(1)
pool.close()
pool.join()

###########################################################################################
#第二步，将生成的结果进行排序。直接放在内存列表中，
sort_rl=sorted(mul_list,key=lambda x:(x[0],x[1],x[2],x[3],x[4],x[5])) 

#for i  in sort_rl:
#    print (i)



###########################################################################################
#3 读取第二步产生的有序的结果，按照实际符合清空数目分开，存储在00中。
out_path=os.path.join(resultpath,out_file)

if not os.path.exists(out_path):
    os.makedirs(out_path)

for each_res in sort_rl:
    for i in range(len(time_list)+1):
        if int(each_res[0])==i:
            path2=out_path+'/'+str(i)
            fb=open(path2,'a+')
            each_res = [str(i) for i in each_res]
            fb.write(' '.join(each_res))
            fb.write('\n')
            fb.close()

print()
b=time.time()
print ('time-consuming',b-a)















'''
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
out_path=os.path.join(resultpath,out_file)

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

subprocess.call('rm znew.txt',shell=True)
print()
b=time.time()
print ('time-consuming',b-a)


'''












    
    
    
    
