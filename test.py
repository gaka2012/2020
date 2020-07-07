#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 10:13:53 2020

@author: zhangzhipeng
"""

import cv2 
import numpy as np

#进行垂直投影，将传入的灰度图转为黑白二值图，统计出每列的黑色点，然后再将统计结果全部集中在底部显示
def getVProjection(image):
    # 将image图像转为黑白二值图，ret接收当前的阈值，thresh1接收输出的二值图
    ret, thresh1 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)   #https://www.cnblogs.com/yinliang-liang/p/9293310.html  
                                                                    #第一个原图像，第二个进行分类的阈值，第三个是高于（低于）阈值时赋予的新值，第四个是一个方法选择参数
    (h,w)=thresh1.shape #返回高和宽
    a = [0 for z in range(0, w)] #a = [0,0,0,...,0,0]初始化一个长度为w的数组，用于记录每一列的黑点个数  
     
    #记录每一列有多少个黑点。
    for j in range(0,w): #遍历一列 
        for i in range(0,h):  #遍历一行
            if  thresh1[i,j]==0:  #如果该点为黑点
                a[j]+=1          #该列的计数器加一计数
                thresh1[i,j]=255  #记录完后将其变为白色 
          
    for j  in range(0,w):  #遍历每一列
        for i in range((h-a[j]),h):  #从该列应该变黑的最顶部的点开始向最底部涂黑
            thresh1[i,j]=0   #涂黑
    
    cv2.imshow('Vimage',thresh1)
    
    
'''
img1 = cv2.imread('new2.png',0) #read picture; default no alpha; but you can set 1 to include alpha.('new.png',1)
                                #set 0 means read gray picture

print (img1.shape)

a = [1,2,3,4,5,6]
b = np.reshape(a,(2,3,1))
#c = np.array([[0,0,0]])
#d = np.insert(b, 2, values=c, axis=0) #插入矩阵，第一个参数是主矩阵，第二个参数是插入的位置，第三个参数是插入的值，第四个参数表示插入行或者列。
print (b)

#bb = b.copy()
#print (bb)
#print (b.shape[0])

for i in range(375):
    for j in range(50):
        print (img1[i][j],end=" ")

    print ()

'''


'''
img1 = cv2.imread('new1.png',0)
#all kinds of filtering
#img1 = cv2.medianBlur(img1, 5) #median filtering ,5 represents choose 5 values and calculate the median.
#img1 = cv2.blur(img1,(3,3)) #mean filtering , ()is the box, 
img1 = cv2.GaussianBlur(img1, (9, 9), 1)

# the parameter in the last are very important
img1 = cv2.adaptiveThreshold(img1,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,77,11)

cv2.imwrite('median.png',img1)  #save file ,input name and data
'''
'''
kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3, 3))
img1 = cv2.morphologyEx(img1, cv2.MORPH_OPEN, kernel1) 

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(6, 6))
img1 = cv2.erode(img1, kernel)
#img1 = cv2.medianBlur(img1, 5)  
cv2.imwrite('median.png',img1)  #save file ,input name and data
#print (img1.shape)
'''


'''
img1 = cv2.imread('new3.png')

img = cv2.imread('new1.png',0) #read picture; default no alpha; but you can set 1 to include alpha.('new.png',1)
                                #set 0 means read gray picture
                                
ret,thresh1 = cv2.threshold(img,230,255,cv2.THRESH_BINARY_INV) #将灰度值大于230的都设置成白色，小于230的都设置成黑色。INV表示与之相反。
                                                               #thresh1是处理后的数据，不要被它的英文迷惑了。
                                                               
cv2.imwrite('two_value.png',thresh1)
image,contours,hierarchy = cv2.findContours(thresh1,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) #二值化后的数据，轮廓的种类(是否有父项等)，是否存储所有的轮廓点，
#输出的结果格式numpy ,list ,numpy
#二值化后的图，与thresh1一样，  列表形式存储的所有的轮廓坐标点、用4个量表示的轮廓属性(Next,Previos,First_child,Parent)

print (image.shape,len(contours),hierarchy.shape)

#print (hierarchy[0][0])
#for i in range(len(contours)):

#    print (contours[i].shape,i)
#img = cv2.drawContours(img,)

cnt = contours[0:800]                            #原始轮廓列表太多了，重新整一下。
#cnt = contours[195]
img = cv2.drawContours(img1,cnt,-1,(0,0,255),3)  #画轮廓，输入底图(在底图上画轮廓，注意底图不能是灰度图，否则显示不了颜色)、轮廓列表、第几个轮廓(-1全画)、颜色、线宽
#cv2.imshow('img',img)

cv2.imwrite('draw_contour.png',img)  #save file ,input name and data

'''

'''
image , contours , hierarchy = cv2.findContours ( binary , cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE )

cv2.drawContours(img,contours,-1,(0,0,255),3) 

#cv2.putText(img, "{:.3f}".format(len ( contours )), (30, 30),cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 1)   
#cv2.imshow("img", img)  
cv2.imwrite("contours.jpg",img)
'''



#opencv官方文档
#https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contours_begin/py_contours_begin.html#contours-getting-started

#img1 = cv2.imread('new2.png')

#print (img1.shape)

#print (img1[0])

#for i in range(375):
#    for j in range(50):
#        print (img1[i][j],end=" ")

#    print ()



'''
#根据一个颜色画图
#a = [235,238,244] #灰白色
#a = [230,236,243]  #灰白色
#a = [219,181,166]   #蓝色
a = [179,146,205]    #红色
#40行3列 z=2的三维矩阵
m = [[a  for _ in range(900) ] for _ in range(900)]
m = np.reshape(m,(900,900,3))
cv2.imwrite("rgb.jpg",m)
#print (m)
'''




img1 = cv2.imread('new2.png') #read picture; default no alpha; but you can set 1 to include alpha.('new.png',1)
                                #set 0 means read gray picture

red = np.array([179,146,205]) #一维向量，(3,)，可以直接用下标来表示输出。
print (img1.shape)

bina = np.ones((img1.shape[0],img1.shape[1])) #生成10行10列的矩阵，元素都是1
for i in range(img1.shape[0]):
    for j in range(img1.shape[1]):
        #print (img1[i,j].shape,img1[i,j]-red,img1[i,j]) #读入的是rgb三原色，是一个一维的向量，可以直接用下标表示，然后减去红色的rgb
        diff = img1[i,j]-red 
        if abs(diff[0])<30 and abs(diff[1])<30 and abs(diff[2])<30:
            bina[i][j]=0 #如果是红色或者是很接近红色的颜色就设为纯白，否则设为黑色。
        else:
            bina[i][j]=255    

print (bina.shape)

lines = cv2.HoughLines(bina, 1, np.pi/180,200)
lines = np.reshape(lines,(lines.shape[0],-1))

for rho, theta in lines:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    cv2.line(img, (x1,y1), (x2, y2), (0,0,255),2)

cv2.imwrite("test.jpg",img1)




