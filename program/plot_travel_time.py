#!/usr/bin/python
# -*- coding:UTF-8 -*-


import os,sys,glob
from obspy.core import UTCDateTime
import csf_phase

#csf1 = csf_phase
#print(type(csf1))
#print(csf1.Phase)



#csf_file = 'CSF-phase-example.txt'
#csf2 = csf_phase.Csf(csf_file) #调用文件中的类，输入文件名。
#print ((csf2.sta_pha()))
#print (csf2.sta_info())        #统计台网-台站，每次添加数字‘1’
#print (csf2.pha_name())
#print (csf2.plot_travel_time())



#csf_file = ['2018-01.txt','2018-02.txt','2018-03.txt','2018-04.txt','2018-05.txt','2018-06.txt','2018-07.txt','2018-08.txt','2018-09.txt','2018-10.txt',
#'2018-11.txt','2018-12.txt','2019-01.txt','2019-02.txt','2019-03.txt','2019-04.txt','2019-05.txt','2019-06.txt']
#csf_file = ['2018-01.txt','2018-02.txt','2018-03.txt','2018-05.txt']
#sta_list = ['WDT','ZHQ','JZG','QCH','PWU','JMG','ZJG','AXI','WCH','CD2','YZP','BAX','MDS','TQU','GZA','HMS'] #李君画图台站列表


csf_file  = ['2018-01.txt']
csf_path  = '/home/zhangzhipeng/software/github/2020/program/report' #存放震相报告的路径
sta_list  = ['AXI']
net_list = ['SC']
test2 = csf_phase.plot_travel_time(csf_path,csf_file,net_list,sta_list) #输入的参数是震相报告的路径，震相报告文件列表，台网列表，台站列表。
test2.plot_figure('figure1') #参数是生成的图像保存位置






