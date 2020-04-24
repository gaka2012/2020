#!/usr/bin/python
# -*- coding:UTF-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import random



#格雷编码返回到二进制编码,输入一个格雷编码基因，返回一个二进制基因。
def gray_decode(bin_num):
    te_str = ''
    an_point = int(bin_num[2])  #初始锚点是左边第一个基因。
    for i in range(3,len(bin_num)):
        te = an_point^int(bin_num[i]) #每次异或完后的基因作为新的锚点
        an_point = te
        te_str+= str(te)
    return bin_num[0:3]+te_str

#二进制吗到格雷编码，输入一个二进制基因，返回一个格雷编码。
def gray_encode(bin_num):
    te_str = ''
    for i in range(len(bin_num)-1,2,-1):
        te = int(bin_num[i])^int(bin_num[i-1])
        te_str=''.join([str(te),te_str])
    return bin_num[0:3]+ te_str         


'''
a= 60
b = bin(a)
print (b)

en = gray_encode(b)
print (en)

test = gray_decode(en)
print (test)
'''

def test(a):
    c =10
    for i in range(len(a)):
        a[i] = a[i]+10
    return c

a = [1,2,3,4,5,6,7,8]
b = test(a)
print (b)
print (a)





    
    
    
    
    
    
    


