#!/usr/bin/python
# -*- coding:UTF-8 -*-

from obspy.core import UTCDateTime

#获取txt中的时间，添加到列表中，格式是UTCdate
m_list = []
fa = open('test.txt','r')
a1 = fa.readlines()
fa.close()
for line in a1:
    mpick = UTCDateTime(line.split()[1])
    m_list.append(mpick)
print(m_list)


#获取FP结果中的结果，
#AXI    -12345 BHZ    ? P0_    ? 20171231 2229  -47.0293 GAU 3.900e-01 0.000e+00 5.274e+00 2.560e+00
fa = open("zday1.txt",'r')
a1 = fa.readlines()
fa.close()
i = 0

for line in a1:
    sp = line.split()
    inum,date,day,sec = sp[4],sp[6],sp[7],sp[8]
    if inum == 'P0_':
        num = i
        i+=1
        print ()
    apick  = UTCDateTime(date+day+sec) #注意对于负的时间是不好用的。
    dtime  = apick-m_list[num]     
    #print (inum,date,day,sec)     
    print (dtime)
