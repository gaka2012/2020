#!/usr/bin/env python 
# -*- coding:utf-8 -*-

import cv2
import numpy as np



#水平投影。
def getHProjection(image):  #输入二值化后的图像，统计每一行中的白色像素点的数量。
    hProjection = np.zeros(image.shape,np.uint8)
    #图像高与宽
    (h,w)=image.shape
    #长度与图像高度一致的数组
    h_ = [0]*h 
    #循环统计每一行白色像素的个数
    for y in range(h):
        for x in range(w):
            if image[y,x] == 255:
                h_[y]+=1
    #绘制水平投影图像
    for y in range(h):
        for x in range(h_[y]):
            hProjection[y,x] = 255
    cv2.imwrite('hProjection.png',hProjection)
 
    return h_
 
def getVProjection(image):
    vProjection = np.zeros(image.shape,np.uint8);
    #图像高与宽
    (h,w) = image.shape
    #长度与图像宽度一致的数组
    w_ = [0]*w
    #循环统计每一列白色像素的个数
    for x in range(w):
        for y in range(h):
            if image[y,x] == 255:
                w_[x]+=1
    #绘制垂直平投影图像
    for x in range(w):
        for y in range(h-w_[x],h):  #这种表达方式使得纵向从底向上显示。
            vProjection[y,x] = 255
    #cv2.imshow('vProjection',vProjection)
    cv2.imwrite('VProjection.png',vProjection)
    return w_
 
 
 

if __name__ == "__main__":

    #将日期那一个横行进行垂直投影，然后将数字切割出来，注意，切割的时候左右边界分别加上5个像素，防止切的太过。
    #报错是因为左右加上的5个像素会使得第一个和最后一个的位置超出边界。
    #切割出来的数字图片的名称包括：1、2、3、4、6、8、9、16、19、20
    '''
    #读入原始图像
    origineImage = cv2.imread('1111.png')  #(232, 1117, 3)
    # 图像灰度化   
    #image = cv2.imread('test.jpg',0)
    image = cv2.cvtColor(origineImage,cv2.COLOR_BGR2GRAY)
        
    # 将图片二值化,自适应二值化效果好很多。
    #retval, img = cv2.threshold(image,170,255,cv2.THRESH_BINARY) 
    img = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,117,8)
    #cv2.imwrite('binary.png',img)
    
    #图像高与宽
    (h,w)=img.shape
    Position = []
    #垂直投影
    V = getVProjection(img) #[227, 224, 189,分界线大约是180左右。
    #print (V)

    
    start = 0
    V_Start = [] #理论上一个start搭配一个end，但是也有最后一个start可能没有搭配的end
    V_End = []
    #根据垂直投影获取垂直分割位置，存储在V_start和V_end中。
    for i in range(len(V)):
        if V[i] > 0 and start ==0: #第一个判断是一个阈值，设定起始点，当白色点数值大于？时将其视为分割的起点。
            V_Start.append(i)
            start = 1
        if V[i] < 1 and start == 1: #第一个判断是结束阈值，设为终点，当白色点数值小于？时，将其视为分割的终点。
            V_End.append(i)
            start = 0
            
    for i in range(1,len(V_End)):
        cut_img = origineImage[0:h,V_Start[i]-5:V_End[i]+5]          
        cv2.imwrite('{}.png'.format(i),cut_img)
    #cut_img = img0[33:265,337:1454] #取出图像，注意仍然是先高度，再宽度，和其原始的shape是一样的。   
    
    '''
    
    '''
    num_list = [1,2,3,4,6,8,9,16,19,20] #有数字的图片名称
    for name in num_list: #将切割后的单独的图片再切割成单独的一个一个的图片，注意仍然是添加了冗余。报错先不用管。
        png_name = str(name)+'.png'
        
        #以下程序将图片分割成上下2部分
        
        #读入原始图像
        origineImage = cv2.imread(png_name)
        # 图像灰度化   
        #image = cv2.imread('test.jpg',0)
        image = cv2.cvtColor(origineImage,cv2.COLOR_BGR2GRAY)
            
        # 将图片二值化,自适应二值化效果好很多。
        #retval, img = cv2.threshold(image,170,255,cv2.THRESH_BINARY) 
        img = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,117,8)
        #cv2.imwrite('binary.png',img)
        
        #图像高与宽
        (h,w)=img.shape
        Position = []
        #垂直投影
        
        H = getHProjection(img) #水平投影。

        start = 0
        H_Start = [] #理论上一个start搭配一个end，但是也有最后一个start可能没有搭配的end
        H_End = []
        #根据垂直投影获取垂直分割位置，存储在V_start和V_end中。
        for i in range(len(H)):
            if H[i] > 0 and start ==0: #第一个判断是一个阈值，设定起始点，当白色点数值大于？时将其视为分割的起点。
                H_Start.append(i)
                start = 1
            if H[i] < 1 and start == 1: #第一个判断是结束阈值，设为终点，当白色点数值小于？时，将其视为分割的终点。
                H_End.append(i)
                start = 0
         
        for i in range(len(H_End)): 
            cut_img = origineImage[H_Start[i]-5:H_End[i]+5,0:w]    #添加冗余，5个像素。    
            cv2.imwrite('{}_{}.png'.format(name,i),cut_img)
    '''
    
    #前2步将数字剪切下来，下面对每张图片挨个进行预处理，先resize,然后二值化，不能反着来，因为resize会改变二值化后的值，使其介于0-255之间。有的图像有噪声，所以开操作一下。
    
    img = cv2.imread('8_0.png',0)
    resize_img = cv2.resize(img, (28,28)) # 调整图像尺寸为28*28
    img = cv2.adaptiveThreshold(resize_img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,117,8) #自适应二值化
    
    #开操作，去除噪声
    kernelX = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    image = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernelX)
    #kernelY = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    #image = cv2.dilate(img, kernelX) 
    #image = cv2.erode(img, kernelY) #腐蚀，输入图像，以及腐蚀的形态，
    #ret, thresh_img = cv2.threshold(resize_img,1,255,cv2.THRESH_BINARY_INV) # 二值化
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            print (image[i,j],end=' ')
        print ('\n')
    cv2.imwrite('000.png',image)    

   

    #以下程序将图片分割成左右2部分。
    '''
    #读入原始图像
    origineImage = cv2.imread('test1.png')  #(232, 1117, 3)
    # 图像灰度化   
    #image = cv2.imread('test.jpg',0)
    image = cv2.cvtColor(origineImage,cv2.COLOR_BGR2GRAY)
        
    # 将图片二值化,自适应二值化效果好很多。
    #retval, img = cv2.threshold(image,170,255,cv2.THRESH_BINARY) 
    img = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,117,8)
    #cv2.imwrite('binary.png',img)
    
    #图像高与宽
    (h,w)=img.shape
    Position = []
    #垂直投影
    V = getVProjection(img) #[227, 224, 189,分界线大约是180左右。
 
    
    start = 0
    V_Start = [] #理论上一个start搭配一个end，但是也有最后一个start可能没有搭配的end
    V_End = []
    #根据垂直投影获取垂直分割位置，存储在V_start和V_end中。
    for i in range(len(V)):
        if V[i] > 170 and start ==0: #第一个判断是一个阈值，设定起始点，当白色点数值大于？时将其视为分割的起点。
            V_Start.append(i)
            start = 1
        if V[i] < 170 and start == 1: #第一个判断是结束阈值，设为终点，当白色点数值小于？时，将其视为分割的终点。
            V_End.append(i)
            start = 0
            
    for i in range(2):
        cut_img = origineImage[0:h,V_End[i]:V_Start[i+1]]        
        cv2.imwrite('{}.png'.format(i),cut_img)
    #cut_img = img0[33:265,337:1454] #取出图像，注意仍然是先高度，再宽度，和其原始的shape是一样的。         
    '''
            
            
    '''        
    #分割行，分割之后再进行列分割并保存分割位置
    for i in range(len(H_Start)):
        #获取行图像
        cropImg = img[H_Start[i]:H_End[i], 0:w]  #对二值化后的图片分割，先分割行或列，再分割列或行。
        #cv2.imshow('cropImg',cropImg)
        
        #对切割后的图像进行垂直投影，返回与宽度一致的列表。
        W = getVProjection(cropImg)
        Wstart = 0
        Wend = 0
        W_Start = 0
        W_End = 0
        for j in range(len(W)):
            if W[j] > 0 and Wstart ==0:  #第一个判断是一个阈值，设定起始点，当白色点数值大于？时将其视为分割的起点。
                W_Start =j
                Wstart = 1
                Wend=0
            if W[j] <= 0 and Wstart == 1:   #第一个判断是结束阈值，设为终点，当白色点数值小于？时，将其视为分割的终点。
                W_End =j
                Wstart = 0
                Wend=1                      #当有一个终点时，即有一个完整的开始和结束，Wend=1
            if Wend == 1:
                Position.append([W_Start,H_Start[i],W_End,H_End[i]])
                Wend =0
                
    #根据确定的位置分割字符
    for m in range(len(Position)):  
        cv2.rectangle(origineImage, (Position[m][0],Position[m][1]), (Position[m][2],Position[m][3]), (0 ,229 ,238), 1) #在原始彩图上画矩形框，输入左上、右下、颜色、宽度。
    cv2.imshow('image',origineImage)                                                                                    #貌似是先宽度再高度。
    cv2.waitKey(0)
    '''








