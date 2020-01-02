#!/usr/bin/python
# -*- coding:UTF-8 -*-


import sys

in_put   = sys.argv

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
start_time = in_put[start_time_index+1]
end_time   = in_put[start_time_index+2]
lat_left   = in_put[lat_index+1]
lat_right  = in_put[lat_index+2]
lon_left   = in_put[lon_index+1]
lon_right  = in_put[lon_index+2]

mag        = in_put[mag_index+1]
sta        = in_put[sta_index+1]
print (sta)
