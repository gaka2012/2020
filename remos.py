_authors__ ="Angel Bueno Rodriguez, Alejandro Diaz Moreno and Silvio De Angelis"
__email__ = "angelbueno@ugr.es"

"""REMOS algorithm main script. 

This script allows the user to use the REMOS segmentation algorithm as described in:

Recursive Entropy Method of Segmentation for Seismic Signals. A. Bueno1, A. Diaz-Moreno, S. De Angelis, 
C. Benitez and J.M.Ibanez. Seismological Research Letters (SRL). 

It is assumed that the seismic data streams has been previously processed and stored in an correct format 
(e.g: miniseed). In practice, if the data can be stored in NumPy format, it can be processed in Python. 

This script requires that `obspy`, `scikits`, `scipy` to be installed within the Python environment you are running 
this algorithm in.

This file can also be imported as a module and contains the following functions:

    * clean_data - returns the segmented candidates that are above or below a threshold.
    * short_term_energy - returns the short term energy of a signal.
    * energy_per_frame - returns an array containing the energy of the framed signal. 
    * plot_signals - Plots a signal, which can be saved or not. 
    * do_sta_lta - Perform a STA/LTA step to obtain the activation time. 
    * run_remos - Runs the REMOS segmentation algorithm. returns the segmented candidates. 
    
"""

import copy,os,sys
import numpy as np
import obspy
#from scikits.talkbox import segment_axis  # it will be deprecated in a future.
import segment_axis
import scipy
from obspy.core import UTCDateTime
from obspy.core import read
from scipy.stats import entropy as sci_entropy
import matplotlib.pyplot as plt
import matplotlib
from obspy.signal.trigger import trigger_onset, recursive_sta_lta

#比如标准答案文件
#改程序用于将某个文件中某行的时间存储成列表，保留原始格式，比如2019-06-19T05:57:17.558000Z
def ge_time_list(time_file,line_num):
    time_list=[]
    with open(time_file,'r') as fr:
        for line in fr:
            part=line.split()
            time_list.append(part[line_num])
            
    return time_list

#该程序与上一个程序配套使用，读取filterpicker产生的结果，看看结果时间与答案是否相符，相符计为1,否则计为0
def is_not_eq(pick_file,an_time_list):
    is_not=[]
    num   =0
    with open(pick_file,'r') as fr:
        for line in fr:
            part=line.split()
            s1,s2,s3=part[6],part[7],part[8]
            picktime=UTCDateTime(s1+' '+s2+' '+s3)
            for m in an_time_list:
                an_time=UTCDateTime(m)
                #如果时间误差是2秒以内，则计为正确。
                if    -2<=(an_time-picktime)<=2:
                    num+=1
            if num==0:
                is_not.append(0)
            else:
                is_not.append(1)
                num=0
    return is_not
#test=ge_time_list('mer2.txt',3)
#if_not=is_not_eq('zday1.txt',test)
#print (if_not)

#1 读取某个seed文件，将里面的trace排序，然后合并，线性插值(感觉并没有什么卵用)
def seed2sac (filename,out_name):
    alltime=[]#创建一个列表用来存储的长度
    f1=read(filename)
    #根据starttime排序
    f1.sort(['starttime'])
    #合并
    f1.merge(method=1,fill_value='interpolate',interpolation_samples=-1)
    #print (len(f1))
    f1[0].write(out_name,format='SAC')
#seed2sac('SC.HWS.00.BHZ.20190618000000.mseed','mer222.sac')

#将filterpicker中拾取出的点数数据转换为remos可以使用的数据格式，由于filterpicker只有开始，没有结束，因此加上了1000个点结束。
def filter_out(input_file):
    filter_file=input_file
    filter_list=[]
    with open(filter_file,'r')as fr:
        for line in fr:
            part         = line.split()
            filter_begin = int(float(part[0]))
            filter_end   = filter_begin+1000 #因为没有end,所以人为加上1000个点。
            tem=[filter_begin,filter_end]
            filter_list.append(tem)
    filter_list=np.array(filter_list) 
    return filter_list   #最后生成的列表，与remos原始的数据格式一样。

               #有3个pick
def clean_data(candidate_segmented, fm=100.0, snr_thr=30.0, min_duration=10.0):
    """

    Parameters
    ----------
    candidate_segmented: Numpy Array
       dddddddd	++++++++++++++++++++++++++++++++++++---- A numpy array containing the segmented candidates from REMOS.
    fm: float
        The sampling frequency of the candidates.
    snr_thr: float
        The minimum SNR requirement for the candidate
    min_duration: float
        Duration (in seconds) to be considered.

    Returns
    -------
    list
        A list containing just the selected, final candidates.
    """

    new_candidate = copy.copy(candidate_segmented)
    not_wanted = set()

    for k, m in enumerate(candidate_segmented):
        ff = m[0] #ff是切出来的数据
        ff = ff - np.mean(ff) #数据去均值
        #根号下 1/2000×(每个数的平方和)
        upper_root = np.sqrt(1 / float(len(ff[0:2000]) * np.sum(np.power(ff[0:2000], 2))))

        #根号下  1/倒数2秒的长度乘以每个数的平方和
        noise_root = np.sqrt(1 / float(len(ff[-int(2.0) * int(fm):]) * np.sum(np.power(ff[-int(2.0) * int(fm):], 2))))
        
        #信噪比计算公式
        print(upper_root,noise_root)
        snr = 10 * np.log(upper_root / noise_root)
        print (snr)
        samples = len(ff)
        
        if (len(ff) / float(fm)) <= min_duration:
            not_wanted.add(k)
        elif np.abs(snr) <= snr_thr:
            not_wanted.add(k)
        else:
            pass

    # we need to iterate over new_candidate to avoid mistakes
    return [m for e, m in enumerate(new_candidate) if e not in not_wanted]


def short_term_energy(chunk):
    """Function to compute the short term energy of a signal as the sum of their squared samples.
    Parameters
    ----------
    chunk: Numpy array
        Signal we would like to compute the signal from
    Returns
    -------
    float
        Containing the short term energy of the signal.
    """
    return np.sum((np.abs(chunk) ** 2) / chunk.shape[0])





def energy_per_frame(windows):
    """It computes the energy per-frame for a given umber of frames.

    Parameters
    ----------
    windows: list
        Containing N number of windows from the seismic signal
    Returns
    -------
    Numpy Array
        Numpy matrix, size N x energy, with N the number of windows, energy their associate energy.
    """
    out = []
    for row in windows: #row代表每一行
        out.append(short_term_energy(row)) #每个值的平方和再处以5
    return np.hstack(np.asarray(out))


def compute_fft(signal, fm):
    """Function to compute the FFT.
    Parameters
    ----------
    signal: Numpy Array.
        The signal we want to compute the fft from.
    fm: float
        the sampling frequency
    Returns
    -------
    Y: Numpy Array
        The normalized fft
    frq: Numpy Array
        The range of frequencies
    """
    n = len(signal)  # length of the signal
    k = np.arange(n)
    T = n / fm
    frq = k / T  # two sides frequency range
    frq = frq[range(n / 2)]  # one side frequency range
    Y = np.fft.fft(signal) / n  # fft computing and normalization
    Y = Y[range(n / 2)]
    return Y, frq


def plot_signals(signal, label, save=None):
    """Function to plot the a seismic singal as a nu,py array.
    Parameters
    ----------
    signal Numpy Array
         Contains the signals we want to plot
    label: str
         A string containing the label or the signal (or the graph title).
    save: str, optional
        A string containing the picture name we want to save
    -------
    """

    plt.figure(figsize=(10, 8))
    plt.title(label)
    plt.plot(signal)

    if save is not None:
        plt.savefig(save + ".png")
    else:
        plt.show()

    plt.close()


def do_sta_lta(reading_folder, nsta, nlta, trig_on, trig_of, filtered=True, bandpass=None):
    """Function to perform STA/LTA on a given trace and recover the data. It assumess a bandpass filter.
    Other functions/parameters to process the data are encouraged.
    Parameters
    ----------
    reading_folder: basestring
         A string indicating the location amd the name of out data (e.g; data/MV.HHZ.1997.01)
    nsta: float
        Length of short time average window in samples
    nlta: float
        Length of long time average window in samples
    trig_on:
        Frequency to trig on the STA/LTA.
    trig_of:
        Frequency to trig off the STA/LTA.
    filtered: bool, optional
        If we want to apply a bandpass filter to the trace or not.
    bandpass: list, optional
        A list containing the flow and fhigh in which we want to bandpass the signal

    Returns
    -------
    original: Stream Obspy Object.
        The copy of the original data,
    st: Stream Obspy Object.
        The processed, filtered stream.
    data: Numpy Array
        The processed data in array format.
    on_of: Numpy Array
        The number of detections and on/of onsets, size N x 2, with N the number of detections.
    """

    st = obspy.read(reading_folder)  # 1 trace within 1 Stream
    original = st.copy()  # we copy the stream in memory to avoid potential NUmnpy missplacements.
    fm = float(st[0].stats.sampling_rate) #采样频率

    if filtered:  #默认filtered是True,因此这个if是直接运行的。
        st = st.filter("bandpass", freqmin=bandpass[0], freqmax=bandpass[1])

    data = st[0]
    data = np.asarray(data) #将数据转换为numpy格式，滤波后的

    #cft = recursive_sta_lta(data, int(nsta * fm), int(nlta * fm))#返回特征函数
    cft=0
    #on_of = trigger_onset(cft, trig_on, trig_of) #根据特征函数和阈值1,阈值2来挑选picks
    on_of=filter_out('num.txt')
    return original, st, data, cft, on_of 
    #返回没有进行任何处理的原始数据，返回滤波后的数据，返回滤波后的numpy格式的data，返回sta/lta的特征函数，返回picks

         #原始数据，滤波后的numpy,numpy matrix,
def run_remos(stream, data, on_of, delay_in, durations_window, epsilon=2.5, plot=False, cut="original"):
    """ Function that executes the main segmentation. Additional pre-processing steps might be required. Please, refer
    to the main manuscript, or github.com/srsudo/remos for additional examples.
    Parameters
    ----------
    stream: Stream Obspy
        The original Stream Obspy object
    data: Numpy Array
        The PROCESSED data from the STA/LTA method
    on_of:
        The numpy matrix, size nsamples x 2, containing the timing
    delay_in: float
        The offset defined to cut from the estimated number of windows
    durations_window: list
        An array containing [W_s, W_d] the window search duration and the minimum window.
    epsilon: float, optional
        The threshold value for the entropy
    plot: bool, optional
        True if we want to plot eacf ot the segmented signals. Be vareful for long streams (>25 min)
    cut:string, optional
        "original" to cut from the main trace, or "processed" to cut from the STA/LTA filtered trace.
    Returns
    X: list
        A list containing the [signal, FI_ratio, start, end]
    -------
    """
    #前3行，复制原始stream,去线性趋势，高通滤波
    # we make a copy in memory of the original array
    array_original = stream[0].copy()  
    # mean removal, high_pass filtering of earth noise background
    array_original = array_original.detrend(type='demean')
    array_original.data = obspy.signal.filter.highpass(array_original.data, 0.5, df=array_original.stats.sampling_rate)

    # plot_signals(array_final[0:10000], "DATA-REMOS-final")
    #采样频率
    fm = float(array_original.stats.sampling_rate)
    X = []

    window_size = durations_window[0]  #10
    search_window = durations_window[1] #200
    #滤波后的numpy减去均值
    data = data - np.mean(data)   
    processed_data = data.copy()

    # use the percentile to reduce background
    umbral = np.percentile(data, 80)  #求取80%处的数值 ==646
    data = (data / float(umbral) * (data > umbral)) + (0 * (data <= umbral))
    print ('umbral===',umbral)
    #return data
    test=ge_time_list('mer2.txt',3)
    if_not=is_not_eq('zday1.txt',test) #返回的结果是一串非1即0的数字，1代表地震，0代表非地震。
    
    total=len(on_of)
    for m, k in enumerate(on_of):
        bar_num=m+1
        percent=bar_num/total
        #sys.stdout.write("\r{0}{1}".format("#"*10 , '%.2f%%' % (percent * 100)))
        #sys.stdout.flush()
        #m表示索引值，0,1,2,3,等   k是结果 [19049 20026]
        #print (m,k)
        # c = c + 1
        #start和end代表的是输入的开始和结束
        start = int(k[0])
        end = int(k[1])
        #x_0是pick的开始减去4.5×100
        x_0 = int(start - delay_in * fm)
        x_1 = np.abs(x_0 - int(start - delay_in * fm + end + search_window * fm))
        # x_1 = np.abs(x_0-int(start+search_window*fm))
        #print ('test==',x_0,x_1)
        #自己添加的只经过滤波以及去均值后的数据。
        processed_candidata= np.asarray(processed_data[x_0:x_1])
        #经过去噪声之后的数据
        selected_candidate = np.asarray(data[x_0:x_1])
        #降维度(本来就是一维的)，并按照window_size的大小分割
        ventanas = segment_axis.segment_axis(selected_candidate.flatten(), int(window_size * fm), overlap=0)
        #print (len(ventanas))
        #返回一个列表或者是一维numpy,里面是每一个window_size长度的数据的平方和除以长度
        energy_ventanas = energy_per_frame(ventanas) #当输入的参数是10 80时，长度为9
        
        #print(len(energy_ventanas))
        #这一步是为了求和，归一化做准备
        total_energy = np.sum(energy_ventanas)
        #返回一个列表或者是一维numpy,里面是每一个window_size长度的数据的平方和除以长度/总的能量，这一步的归一化其实有点多余，因为sci_entropy会自动归一化。
        loq = energy_ventanas / float(total_energy)
        #求熵如果小于epsilon，则是地震。
        #画图  首先是波形图(滤波并减去均值)processed_candidata 然后是经过去噪声后的数据图selected_candidate，然后是能量图energy_ventanas 以及熵值sci_entropy(loq)
        plot_energy=False
        if plot_energy:
            
            
            plt.figure(figsize=(25,15))
            ax0=plt.subplot(3,1,1)
            plt.plot(processed_candidata)
            name0=str(m+1)+'data'
            plt.title(name0,fontsize=24,color='r')

            ax=plt.subplot(3,1,2)
            plt.plot(selected_candidate)
            name1=str(m+1)+'test'  #第一张图的标题
            plt.title(name1,fontsize=24,color='r')
            
            ax1=plt.subplot(3,1,3)
            plt.plot(energy_ventanas)  
            entropy_title=sci_entropy(loq) #根据能量计算熵值
            name=str(m+1)+ 'test.png'
            plt.title(str(entropy_title),fontsize=24,color='r') #将熵值作为第二张子图的标题
            plt.savefig(name)
            plt.close()
            #如果是地震
            if if_not[m]==1:
                out_path='img/eq' #将画完的图移动到改文件加下。
            elif if_not[m]==0:
                out_path='img/noise'
            os.system('mv %s %s'%(name,out_path)) 
        
        if sci_entropy(loq) < epsilon:
            #print(np.argmin(loq))#找到最小能量值索引
            cut_me = int(np.argmin(loq) * window_size * fm + delay_in * fm)
            potential_candidate = array_original[x_0:cut_me + x_0]
            
            duration_candidate = potential_candidate.shape[0] / float(fm)
            #print (duration_candidate)
            if duration_candidate < 5.0:
                # By doing this, we erase those windows with small durations
                pass
                print ('能量最小值出现在前5秒中')

            else:
                #                       这个数据是复制的原始数据。经过了去均值以及滤波处理了
                potential_candidate = potential_candidate - np.mean(potential_candidate)
                #这个时候的数据就是已经去掉了能量最小值之后的数据了。然后继续切片长度是5s                
                ventanas_ref = segment_axis.segment_axis(potential_candidate.flatten(), int(5.0 * fm), overlap=0)
                #print (ventanas_ref.shape) #(8, 500)
                try:
                    dsai = int((ventanas_ref.shape[0] / 2.0))
                    
                except:
                    dsai = 0
	
                try: #后半部分的能量和除以前半部分的能量和。
                    down_windows = energy_per_frame(ventanas_ref[0:dsai]) #前半部分
                    upper_windows = energy_per_frame(ventanas_ref[dsai:dsai + dsai]) #后半部分
                    #print (upper_windows)
                    ratio = np.round(np.sum(np.asarray(upper_windows)) / np.sum(np.asarray(down_windows)), 2)
                except:
                    ratio = np.inf #一个无限大的数
                    pass
                #print(ratio)
                if ratio < 0.15: #后半部分的能量除以前半部分
                    #print('ratio is too low')
                    # In this case, long-segmentation, re-cut.
                    try:
                        #找到后半部分能量最小的前2个的索引值，然后按照索引值排序，找到索引值最小的？
                        ind = np.sort(np.argpartition(upper_windows, 2)[:2])[0]
                        #print (ind)
                    except:
                        # print "on exception"
                        ind = upper_windows.shape[0]
                    #
                    min_duration = (down_windows.shape[0] + ind) * 5.0 * fm
                    cut_me = int(min_duration)

            if cut == "original": #返回原始数据或者是滤波后的数据。
                selected_candidate = array_original.data[x_0:cut_me + x_0]
                X.append([selected_candidate, ratio, x_0, cut_me + x_0])
            else:
                selected_candidate = processed_data[x_0:cut_me + x_0]
                X.append([selected_candidate, ratio, x_0, cut_me + x_0])

            # lets plot
            if plot:
                plt.figure()
                plot_signals(selected_candidate, label="SEGMENTED")
        else:
            pass
        
    return X
    

def visualize_segmentation(data, positions):
    """Function to plot the data and visualize the result of segmentation, over the real trace.
    Parameters
    ----------
    data: Numpy array
        The seismic signal as a Numpy array we would like to visualize.
    positions:
        A list containing the on_off triggering data.
    Returns
    -------
    """
    plt.figure(figsize=(20, 8))
    ax = plt.subplot()
    segmented_on = positions[:, 0]
    segmented_off = positions[:, 1]
    plt.title("VISUALIZATION OF SEGMENTATION RESULTS")
    plt.plot(data)
    ymin, ymax = ax.get_ylim()
    plt.vlines(segmented_on, ymin, ymax, color='green', linewidth=2, linestyle='solid')
    plt.vlines(segmented_off, ymin, ymax, color='magenta', linewidth=2, linestyle='dashed')
    plt.savefig('sta_lta_continua.eps', format='eps', dpi=300)
    plt.xlim((0, data.shape[0]))  # set the xlim to left, right
    plt.show()
