#!/usr/bin/python
# -*- coding:UTF-8 -*-

import time,os
import subprocess
import numpy as np
import threading
from multiprocessing import Pool

#修改2.生成的结果存放的位置，result2

#/mnt/bashi_fs/filterpicker/picker
program_path = '/home/zhangzhipeng/software/filterpicker/picker'  #程序所在的路径，需要先进入路径才运行    
input_data   = 'mer2.sac'                                         #.带入反演的数据名称 mer1.sac
out_path     = '/home/zhangzhipeng/software/filterpicker/picker/mer2test'  #生成的结果的文件夹


if not os.path.exists(out_path):
    os.makedirs(out_path)

def mulfilter(item):
    (name2,filterwindow1,longt,tup1,t1,t2)=item
    out_file=out_path+'/'+name2
    subprocess.call('cd %s;./picker_func_test %s %s  %s %s %s %s %s' %(program_path,input_data,out_file,filterwindow1,longt,tup1,t1,t2),shell=True)
                            #要修改存储结果的文件就改这里
parameter=[]#传递给多线程的参数列表
list1=[] #存放6个参数的临时列表(第一个参数是文件名)
start=time.time()
filterwindow1=300
#4个参数的取值范围
for longt in range(240,242,2):  #1 280-301 15-32 6-20 6-20
    for tup1 in range(30,31): #300 300 15 10 15 #300 300 5-15 #注意不要小于5
        for t1 in np.arange(5,6,1):
            for t2 in np.arange(5,15,1):
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


















