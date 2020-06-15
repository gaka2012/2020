#!/usr/bin/python
# -*- coding:UTF-8 -*-


import cut_data as cd
import glob,os
from obspy.core import stream
from obspy.core import read

#寻找数据为0的z分量的数据，最大值和最小值均为0,直接删掉,并进行滤波处理,并重新计算snr.
sac_list = glob.glob('/home/zhangzhipeng/software/github/2020/program/sac_data/*.BHZ*')
os.chdir('/home/zhangzhipeng/software/github/2020/program/sac_data')
for sac_file in sac_list:
    sac_name = os.path.basename(sac_file) #SC.AXI_20180101062906.BHZ.sac
    st = stream.Stream()
    st+= read(sac_file)
    s  = st[0].stats.sac
    #如果最大和最小值为0,则删除数据
    if s.depmax==0 and s.depmin==0:
        print (sac_file)   
        os.system('rm %s'%(sac_name)) 
    #对于正常的数据，则进行滤波、去均值等
    else:
        #滤波，去均值
        st[0].detrend('demean').detrend('linear').taper(max_percentage=0.05, max_length=10.)
        st[0]=st[0].filter('bandpass',freqmin=1,freqmax=15) #带通滤波
        
        #重新计算信噪比
        s = st[0].stats.sac
        r_time = UTCDateTime(year=s.nzyear,julday=s.nzjday,hour=s.nzhour,minute=s.nzmin,second=s.nzsec,microsecond=s.nzmsec*1000) #>    参考时刻
        p_time = r_time+s.a #P波到时
        snr = cd.cal_SNR(st[0],p_time,10,4)
        s.t9 = snr

        #保存文件
        st[0].write(sac_name,format='SAC')
    st.clear()





'''
print ('我是分界线')

#画单个文件的z分量数据图，图名是北京时间的发震时刻-震级-震中距-snr
sac_list = glob.glob('/home/zhangzhipeng/software/github/2020/program/sac_data/*.BHZ*')
save_fig = '/home/zhangzhipeng/software/github/2020/program/sac_figure'
os.chdir(save_fig)
for sac_name in sac_list:
    print (sac_name)
    cd.plot_sac(sac_name)
    #os.system('mv *.png %s'%(save_fig))




#画信噪比分布图,第一步统计文件中z分量信噪比，形成列表。
#修改数据存放位置以及2个台站名称

sta_list = ['AXI','BAX','CD2','GZA','HMS','JMG','JZG','MDS','PWU','QCH','TQU','WCH','YZP','ZJG']
for i in sta_list:
    sac_list = glob.glob('/home/zhangzhipeng/software/github/2020/program/sac_data/*'+i+'*.BHZ*')
    snr_list = []
    for sac_name in sac_list:
        st = stream.Stream()
        st+= read(sac_name)
        s  = st[0].stats.sac.t9
        if s!=-1234:
            #print (s)
            snr_list.append(s)
        st.clear()

    fig_name = 'SC-'+i+'-SNR'
    cd.plot_num_distribution(snr_list,fig_name,8,10,'snr','num')

'''

   










