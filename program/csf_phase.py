



import os,sys,glob,re
from obspy.core import UTCDateTime
import matplotlib.pyplot as plt
import numpy as np

'''
  CSF attributes & headline infomation 
  HBO Net_code date time Epi_lat Epi_lon Epi_depth Mag_name Mag_value Rms Qloc Sum_stn Loc_stn Epic_id Source_id Eq_type Location_cname
  HEO Auto_flag Event_id Sequen_name Depfix_flag M M_source SPmin Dmin Gap_azi Erh Erz Qnet Qcom Sum_pha Loc_pha FE_num FE_sname
  HMB Mag_name Mag_val Mag_gap Mag_stn Mag_error
  HPB Net_code Sta_code Chn_code Clarity Wsign Phase_name Weight Rec_type Phase_time Phase_time Resi Distance Azi Amp Period Mag_name Mag_val
  ...
'''

class DBO(object):
  def __init__(self,net, date, time, epi_lat, epi_lon, epi_dep, mag_value,mag_type,*args):
              # *rms, *qloc, *sum_stn, *loc_stn, *eqic_id, *source_id, *eq_type, *location_cname):
    self.net  = net
    self.date = date
    self.time = time
    self.epi_lat = epi_lat
    self.epi_lon = epi_lon
    self.epi_dep = epi_dep 
    self.mag_value = mag_value
    self.mag_type  = mag_type
    self.eq_ot = str(UTCDateTime(self.date+self.time))
     
  def get_ot(self):
    ot = self.date + self.time
    ot = UTCDateTime(ot)
    return ot

  
class DEO(object):
  def __init__(self,auto_flag, event_id,*args):
               #     *Sequen_name, *Depfix_flag, *M, *M_source, *SPmin, *Dmin, *Gap_azi,
               #     *Erh, *Erz, *Qnet, *Qcom, *Sum_pha, *Loc_pha, *FE_num, *FE_sname):
    self.auto_flag = auto_flag
    self.event_id  = event_id
  
  def get_event_id(self):
    return self.event_id


class DMB(object):
  def __init__(self, Mag_name):
    # *Mag_name,*Mag_val, *Mag_gap, *Mag_stn, *Mag_error):
    self.Mag_name = Mag_name


class DPB(object):
  def __init__(self, Net_code, Sta_code, Chn_code, Phase_name,Phase_date, Phase_time, Distance,azi):
              #       *Clarity, *Wsign, *Weight, *Rec_type, *Resi, *Distance, 
              #       *Azi, *Amp, *Period, *Mag_name, *Mag_val):
    self.Net_code = Net_code.replace(' ','')
    self.Sta_code = Sta_code.replace(' ','')
    self.Chn_code = Chn_code.replace(' ','')
    self.Phase_name = Phase_name.replace(' ','')
    self.Phase_date = Phase_date
    self.Phase_time = Phase_time
    self.Distance   = Distance
    self.azi        = azi
    #self.Phase_ot = str(UTCDateTime(self.Phase_date + self.Phase_time))
    
    # self.Traval_time = str( UTCDateTime(self.Phase_ot) - UTCDateTime(self.eq_ot))
 
  def get_phase_ot(self):
    phase_ot = self.Phase_date + self.Phase_time
    phase_ot = UTCDateTime(phase_ot)
  
  def get_phase_dict(self):
    phase_dict = {}
    key = self.Phase_name  
    phase_dict[key][1].append([phase_ot])   
    

class Phase(object):
  def __init__(self, DBO, DEO, DMB, *DPB):
     self.dbo = DBO
     self.deo = DEO
     self.dmb = DMB
     self.dpb = DPB   

class Csf(object):
  def __init__(self, csf_path,csf_file,net_input,sta_input):
    self.csf_path = csf_path
    self.csf_file = csf_file
    self.net_in   = net_input #列表，存储要画的台网
    self.sta_in   = sta_input #列表，存储要画的台站
    phase_dict = {}
    
    for name in self.csf_file: #读取输入的文件名称列表，一个个的读震相文件。
        name = os.path.join(self.csf_path,name)
        fo = open(name,'r',encoding='gbk'); 
        lines = fo.readlines(); 
        fo.close()
        for i,line in enumerate(lines):
            code_id = line[0:3]

            if code_id == 'DBO' :
      #DBO YN 2019-07-03 04:27:53.43  24.502   98.905  14 ML    2.0好  0.133 2   0  17 53 53 eq 云南龙陵        
                net1     = line[3:6].replace(' ','')
                eq_date  = line[7:18]
                eq_ot    = line[18:30]
                eq_lat   = line[30:38]
                eq_lon   = line[38:47]
                eq_dep   = line[47:51]
                mag_type = line[51:54].replace(' ','')
                mag  = line[57:61 ]
        
                dbo1 = DBO(net1,eq_date,eq_ot,eq_lat,eq_lon,eq_dep,mag,mag_type)

                key_dbo = eq_date + eq_ot
                key_dbo = str(UTCDateTime(key_dbo))
 
                dpb_list = []
                dpb_list.clear()

            if code_id == 'DEO' :
      # DEO C YN.201907030427.0001                      0  1.2 ML                74.8   0.2   1.1 0 0      30  297     
                at_flag = line[4:5]
                evt_id  = line[6:27]
                deo1 = DEO(at_flag, evt_id)
            #phase_dict.setdefault(dbo1,[2]).append(deo1)
       
            if code_id == 'DMB':
                mag_type = line[4:7]
                dmb1 = DMB(mag_type)
        #phase_dict.setdefault(dbo1,[3]).append(dmb1)

            if code_id == 'DPB':
        # Net_code, Sta_code, Chn_code, Phase_name,Phase_date, Phase_time

                pha_net = line[3:6].replace(' ','')
                pha_sta = line[9:13].replace(' ','')
                pha_chn = line[13:17]
                pha_name= line[25:29]
                pha_date= line[34:45]
                pha_time  = line[45:57]
                pha_dis = line[65:72]
                pha_azi = line[72:80]
                
                if pha_sta in self.sta_in and pha_net in self.net_in: #如果台站在输入的台站列表中，则存入字典。
                    dpb1 = DPB(pha_net, pha_sta, pha_chn, pha_name, pha_date, pha_time, pha_dis,pha_azi)
                    phase_dict.setdefault(dbo1,[]).append(dpb1)   #每一个dbo(事件) 对应很多个地震台站到时。
                    dpb_list = phase_dict[dbo1]
            #phase1 = Phase(dbo1, deo1, dmb1, dpb_list)        
            self.phase_dict = phase_dict
  
  
      #print (key.net,key.eq_ot,len(value)) 
  
     #统计台网/台站，每次添加数字‘1’
  def sta_info(self):        
    sta_dict = {}
    for key in self.phase_dict:
        for i, pha_line in enumerate(self.phase_dict[key]):
           net = (pha_line.Net_code).replace(' ','')
           sta = (pha_line.Sta_code).replace(' ','')
           key = net + '-' + sta 
           sta_dict.setdefault(key,[]).append('1') 
    self.sta_dict = sta_dict
    return sta_dict

  #统计震相 每一个后面添加数字‘2’
  def pha_name(self):
    pha_name_dict = {}
    for key in self.phase_dict:
        dbo_line = key
        for i,pha_line in enumerate(self.phase_dict[key]):
        #    print(key.eq_ot, key.epi_lat, key.epi_lon, key.mag_value, pha_line.Sta_code, pha_line.Phase_time)
           pha_name = pha_line.Phase_name
           pha_name_dict.setdefault(pha_name,[]).append('2') 
    self.pha_name_dict = pha_name_dict
    for key in pha_name_dict:
       print(key)
    return pha_name_dict
  
  
  def sta_pha(self):
    sta_pha_dict = {}
    for key in self.phase_dict:
       eq_ot = key.eq_ot           #发震时刻
       eq_lat= key.epi_lat
       eq_lon= key.epi_lon
       eq_dep= key.epi_dep
       #print(eq_ot) 
       for i, pha_line in enumerate(self.phase_dict[key]):
           net = pha_line.Net_code.replace(' ','')  #台网代码
           sta = pha_line.Sta_code.replace(' ','')  #台站代码
           key = net + '-' + sta
           chn = pha_line.Chn_code
           pha_name    = pha_line.Phase_name  #震相名称
           pha_name    = pha_name.replace(' ','') #将震相名称中的空格去掉。
           net_sta_pha = key+'-'+pha_name     #台网-台站-震相，作为字典的键值。
           pha_date = pha_line.Phase_date
           pha_time = pha_line.Phase_time
           pha_dis  = pha_line.Distance   #震中距(str)
           pha_ot   = pha_date + pha_time #到时
           
           if not pha_ot.isspace() and not pha_dis.isspace():   #如果到时以及震中距不是空格
               pha_ot            = UTCDateTime(pha_date +' '+ pha_time)  #到时
               pha_traval_time   = pha_ot- UTCDateTime(eq_ot)            #走时(float)
               tem_list          = [float(pha_dis),pha_traval_time]      #临时列表，存储dist，travel_time，作为键值对应数据
               sta_pha_dict.setdefault(net_sta_pha,[]).append(tem_list)
   
    self.sta_pha_dict = sta_pha_dict
    return sta_pha_dict #字典中的键值是台网-台站-震相，对应 其dist，travel-time


class plot_travel_time(object):
  def __init__(self,input_path,input_file,input_net,input_sta):
    self.input_path = input_path
    self.input_file = input_file
    self.input_net  = input_net
    self.input_sta  = input_sta
    #get phase_dict from class Csf
    read_file = Csf(input_path,input_file,input_net,input_sta)
    self.get_phase = read_file.sta_pha()  #台网-台站-震相字典
    self.get_sta   = read_file.sta_info()      #台网-台站字典
    
  #统计字典sta_dict中的台站数量(统计数字1的数量)  
  def sta_number(self):
    sta_list = [] #台站名称(str)，数量(int)
    for key,value in self.get_sta.items(): 
      tem_sta = [key,len(value)] #台站名称，对应的数字1的数量
      sta_list.append(tem_sta)
    self.sta_list = sta_list
    return sta_list   
  
  
  #plot  
  def plot_figure(self,figure_path):
   
    #color_dict = {'Pg':'r','Pn':'k','Sg':'b','Sn':'g'}  #画图时不同震相颜色字典
    color_dict = {'Pg':'r','Pn':'k'}
    plot_path  = figure_path
    #plt.figure(figsize=(25,15)) 
    tem_name = [] #存储台站-震相名称
    new      = []
    total = len(self.sta_number()) #要画的台站数量总数。
    num   = 0
    for sta in self.sta_number(): #每个台站一幅图
      try:
      #用于写进度条
        num+=1
        percent=num/total 
        sys.stdout.write("\r{0}{1}".format("#"*10 , '%.2f%%' % (percent * 100)))
        sys.stdout.flush()
      
        max_dist       = 0 #最大dist,用来作为x轴最大值
        max_travel     = 0 #最大走时，用来作为y轴最大值
        tem_max_dist   = 0
        tem_max_travel = 0
        plt.figure(figsize=(25,15)) 
        for key,value in self.get_phase.items():  #字典，键值是台网-台站-震相，对应 其dist，travel-time
          if '-'.join(key.split('-')[0:2])==sta[0]:
            phase_name = key.split('-')[-1]
            if phase_name in color_dict.keys():   #只画color_dict中有的震相，其他的不画
              #print (key,value)
              tem_dist   = [] #dist列表
              tem_travel = [] #走时列表，画图数据
              for i in value: #value是每个台站对应的很多个震中距和到时组成的列表，i[0]表示震中距，i[1]表示走时。
                if i[1]<0:
                  pass
                else:
                  tem_dist.append(i[0])
                  tem_travel.append(i[1])
              
              color = color_dict[phase_name]  #不同震相不同颜色
              plt.scatter(tem_dist,tem_travel,s=10,c=color,label=phase_name)   
              tem_max_dist   = max(tem_dist)  #最大dist
              tem_max_travel = max(tem_travel)#最大走时，用来设置图的范围
              if tem_max_dist > max_dist:
                max_dist = tem_max_dist
              if tem_max_travel > max_travel:
                max_travel = tem_max_travel    
        plt.xticks(np.arange(0,max_dist+50,50))   
        plt.yticks(np.arange(0,max_travel+10,10)) 
        plt.legend()
        plt.xlabel('dist',fontsize=25)
        plt.ylabel('travel-time',fontsize=25)
        plt.title(sta[0],fontsize=25)
        plt.savefig(sta[0])
        plt.close() 
      except Exception as e:
        print (e,sta[0])     
    os.system('mv *.png %s'%(plot_path))   
    print () #最后输出一下来换行。   
#        tem_name.append(key)
#        new.append(tem_dist)
#        tem_dist = []
    #self.new = new
    #return new
    #print (test)
  


