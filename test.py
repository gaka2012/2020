#!/usr/bin/python
# -*- coding:UTF-8 -*-
'''
使用方法 python test.py -t 2019-09-
必须有的选项 -t -lat -lon 
可选选项     -m -sta -pha
'''

import sys

in_put   = sys.argv

mag_index   = False
sta_index   = False
phase_index = False
for content in in_put:
    if content == '-t':
        start_time_index = in_put.index(content)
    elif content == '-m':
        mag_index = in_put.index(content)
    elif content == '-lat':
        lat_index = in_put.index(content)
    elif content == '-lon':
        lon_index = in_put.index(content)
    elif content == '-sta':
        sta_index = in_put.index(content)
    elif content == '-pha':
        phase_index = in_put.index(content)
start_time = in_put[start_time_index+1]
end_time   = in_put[start_time_index+2]
begin_time  = UTCDateTime(start_time)
end_time    = UTCDateTime(end_time)
lat_min   = in_put[lat_index+1]
lat_max  = in_put[lat_index+2]
lon_min   = in_put[lon_index+1]
lon_max  = in_put[lon_index+2]

if mag_index: #震级，默认是0-10
    mag  = in_put[mag_index+1].split(',')
    mag_min,mag_max = float(mag[0]),float(mag[1])
    print (mag_min,mag_max)
else:
    mag_min,mag_max=0,10
if sta_index:
    sta_list = in_put[sta_index+1].split(',')
    print (sta_list)
if phase_index:
    phase_list = in_put[phase_index+1].split(',')
    print (phase_list)
print (mag_max)















