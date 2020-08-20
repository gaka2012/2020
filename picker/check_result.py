#!/usr/bin/python
# -*- coding:UTF-8 -*-

import os
import subprocess


#读取test.txt中的数据路径和答案,将数据路径赋值给要调用的程序，得到结果，与标准答案进行对比。
fa = open('test.txt')
A  = fa.readlines()
fa.close()

for line in A:
    path,answer = line.split()
    if answer != '-1234':  #说明改事件是个地震，而不是噪声
        subprocess.call('cd /home/zhangzhipeng/software/filterpicker/picker;./picker_func_test mer2.sac /home/zhangzhipeng/software/filterpicker/picker/test00/%s  %s %s %s %s %s' %(name2,filterwindow1,longt,tup1,t1,t2),shell=True)
        print (answer)
