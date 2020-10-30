#!/usr/bin/python
# -*- coding:UTF-8 -*-

import os,time,threading
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np
import pandas as pd
import threading,time 
import matplotlib.pyplot as plt
from data_reader import Config, DataReader, DataReader_test, DataReader_pred, DataReader_mseed


coord = tf.train.Coordinator()
#1. 调用data_reader中的start_threads，对数据进行处理
data_reader = DataReader(data_dir  = './test_data',
                         data_list = './test_data.csv',
                         mask_window = 0.4,
                         queue_size = 200*3,
                         coord = coord)
    

data_process = data_reader.start_threads()
