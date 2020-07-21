#!/usr/bin/python
# -*- coding:UTF-8 -*-

import cv2
import numpy as np
from PIL import Image


global img
global point1, point2

def on_mouse(event, x, y, flags, param):
    global img, point1, point2
    img2 = img.copy()
    if event == cv2.EVENT_LBUTTONDOWN:         #左键点击
        point1 = (x,y)
        cv2.circle(img2, point1, 10, (0,255,0), 5)  #根据圆心的中心和半径画圆
        cv2.imshow('image', img2)
    elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):   #按住左键拖曳
        cv2.rectangle(img2, point1, (x,y), (255,0,0), 5) # 图像，矩形顶点，相对顶点，颜色，粗细
        cv2.imshow('image', img2)
    elif event == cv2.EVENT_LBUTTONUP:         #左键释放
        point2 = (x,y)
        cv2.rectangle(img2, point1, point2, (0,0,255), 5) 
        cv2.imshow('image', img2)
        min_x = min(point1[0], point2[0])     
        min_y = min(point1[1], point2[1])
        width = abs(point1[0] - point2[0])
        height = abs(point1[1] -point2[1])
        cut_img = img[min_y:min_y+height, min_x:min_x+width]
        resize_img = cv2.resize(cut_img, (28,28)) # 调整图像尺寸为28*28
        ret, thresh_img = cv2.threshold(resize_img,127,255,cv2.THRESH_BINARY) # 二值化
        cv2.imshow('result', thresh_img)
        cv2.imwrite('./images/text.png', thresh_img)  # 预处理后图像保存位置

def main():
    global img
    img = cv2.imread('3.jpg')  # 手写数字图像所在位置
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 转换图像为单通道(灰度图)
    cv2.namedWindow('image')  #命名窗口
    cv2.setMouseCallback('image', on_mouse) # 调用回调函数
    cv2.imshow('image', img)
    cv2.waitKey(0)

if __name__ == '__main__':
    #main()
    '''
    img = cv2.imread('0.png')  # 手写数字图像所在位置
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 转换图像为单通道(灰度图)
    resize_img = cv2.resize(img, (28,28)) # 调整图像尺寸为28*28
    #ret, thresh_img = cv2.threshold(resize_img,127,255,cv2.THRESH_BINARY) # 二值化
    img = cv2.adaptiveThreshold(resize_img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,117,8)
    cv2.imwrite('333.png',img)
    
    print (img.shape)
    #print (resize_img.shape)
    '''

    
    
    
    
    
    im = Image.open('333.png')
    data = list(im.getdata())
    result = [(255-x)*1.0/255.0 for x in data] 
    print(result)
    #print (data)
    
    
    
    
    
    
    
    
    
    
