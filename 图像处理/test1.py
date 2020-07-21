#!/usr/bin/env python 
# -*- coding:utf-8 -*-

import cv2
import numpy as np

'''
# 读取图片
rawImage = cv2.imread("new2.png")
# 高斯模糊，将图片平滑化，去掉干扰的噪声
image = cv2.GaussianBlur(rawImage, (3, 3), 0)
# 图片灰度化
image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)



# Sobel算子（X方向）
Sobel_x = cv2.Sobel(image, cv2.CV_16S, 1, 0)
Sobel_y = cv2.Sobel(image, cv2.CV_16S, 0, 1)
absX = cv2.convertScaleAbs(Sobel_x)  # 转回uint8
absY = cv2.convertScaleAbs(Sobel_y)
dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

#image = absX
image = dst

# 二值化：图像的二值化，就是将图像上的像素点的灰度值设置为0或255,图像呈现出明显的只有黑和白
#ret, image = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)    #自动选择阈值进行二值化
ret, image = cv2.threshold(image, 44, 255, cv2.THRESH_BINARY) #第一个原图像，第二个进行分类的阈值，第三个是高于（低于）阈值时赋予的新值，第四个是一个方法选择参数




# 闭操作：闭操作可以将目标区域连成一个整体，便于后续轮廓的提取。
kernelX = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))
image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernelX)


cv2.imwrite('test.png',image)



# 膨胀腐蚀(形态学处理) #通过膨胀连接相近的图像区域，通过腐蚀去除孤立细小的色块。
kernelX = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
kernelY = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 19))
image = cv2.dilate(image, kernelX) 
image = cv2.erode(image, kernelX) #腐蚀，输入图像，以及腐蚀的形态，
image = cv2.erode(image, kernelY)
image = cv2.dilate(image, kernelY)


# 平滑处理，中值滤波,去除噪声
image = cv2.medianBlur(image, 15)




# 查找轮廓
tmp, contours, w1 = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

for item in contours:
    rect = cv2.boundingRect(item)
    x = rect[0]
    y = rect[1]
    weight = rect[2]
    height = rect[3]
    if weight > (height * 2):
        # 裁剪区域图片
        chepai = rawImage[y:y + height, x:x + weight]
        cv2.imshow('chepai'+str(x), chepai)

# 绘制轮廓
image = cv2.drawContours(rawImage, contours, -1, (0, 0, 255), 3)
cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

'''
img = cv2.imread('new1.png',0) #read picture; default no alpha; but you can set 1 to include alpha.('new.png',1)
                                #set 0 means read gray picture
                                
ret,image = cv2.threshold(img,170,255,cv2.THRESH_BINARY_INV) #将灰度值大于230的都设置成白色，小于230的都设置成黑色。INV表示与之相反。
                                                               #thresh1是处理后的数据，不要被它的英文迷惑了。

cv2.imwrite('two2.png',image) #生成2值化后的图，已保存。

'''

'''                                                       
#cv2.imwrite('two2.png',image) #生成2值化后的图，已保存。
      
kernelX = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)) #选择一个形态，这是个矩形，大小是3×3
image = cv2.erode(image, kernelX) #腐蚀，输入图像，以及腐蚀的形态，取形态的最小值。用来去掉小的白点(噪声)


kernelY = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)) #选择一个形态，这是个矩形，大小是3×3
image = cv2.erode(image, kernelY) #腐蚀，输入图像，以及腐蚀的形态，取形态的最小值。用来去掉小的白点(噪声)

kernelZ = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8)) #选择一个形态，这是个矩形，大小是3×3
image = cv2.dilate(image, kernelZ)#膨胀，与腐蚀相反

#kernelA = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 10)) #选择一个形态，这是个矩形，大小是3×3
#image = cv2.dilate(image, kernelA)#膨胀，与腐蚀相反

cv2.imwrite('two.png',image)

'''



#实现了完整的轮廓识别
'''
def rotate(img, angle): #对原始图片进行旋转。
    #旋转图片
    (h, w) = img.shape[:2]  #获得原始图片高，宽
    center = (w // 2, h // 2)  #获得图片中心点
    img_ratete = cv2.getRotationMatrix2D(center, angle, 1) #center表示中间点的位置，-5表示逆时针旋转5度，1表示进行等比列的缩放
    rotated = cv2.warpAffine(img, img_ratete, (w, h))      #img表示输入的图片，仿射变化矩阵，表示变换后的图片大小
    cv2.imwrite('rotate.png',rotated)
    #return rotated

def cut1(img, box):    #经过旋转的图片
    #从轮廓出裁剪图片
    x1, y1 = box[1]  #获取左上角坐标
    x2, y2 = box[3]  #获取右下角坐标
    img_cut = img[y1+10:y2-10, x1+10:x2-10]  #切片裁剪图像
    return img_cut


rectan = cv2.imread('new1.png') #读入原始图，为画矩形做准备。
img0 = cv2.imread('new1.png') #read picture; default no alpha; but you can set 1 to include alpha.('new.png',1)
                                #set 0 means read gray picture

print (img0.shape)
img1 = cv2.imread('new1.png',0)  #读入灰度图，这样是为了保证输入的数据格式是CV_8U,否则findcontours会报错。


red = np.array([180,150,210])
#red = np.array([179,146,205]) #一维向量，(3,)，可以直接用下标来表示输出。

#red = np.array([0,0,0])
#print (img1.shape)

#bina = np.ones((img1.shape[0],img1.shape[1])) #生成10行10列的矩阵，元素都是1
for i in range(img0.shape[0]):
    for j in range(img0.shape[1]):
        #print (img1[i,j].shape,img1[i,j]-red,img1[i,j]) #读入的是rgb三原色，是一个一维的向量，可以直接用下标表示，然后减去红色的rgb
        diff = img0[i,j]-red 
        if abs(diff[0])<40 and abs(diff[1])<40 and abs(diff[2])<40:
            img1[i][j]=255 #如果是红色或者是很接近红色的颜色就设为纯白，否则设为黑色。
            #print (diff)
        else:
            img1[i][j]=0    
            

#腐蚀，去除噪声
kernelX = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
image = cv2.erode(img1, kernelX)

#kernelA = cv2.getStructuringElement(cv2.MORPH_CROSS, (10, 10)) #选择一个形态，这是个矩形，大小是3×3
#image = cv2.dilate(image, kernelA)#膨胀，与腐蚀相反



# 闭操作：闭操作可以将目标区域连成一个整体，便于后续轮廓的提取。
kernelX = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernelX)

#闭操作，比单独腐蚀再膨胀好很多。这2个都是矩形，一个是用来链接横向的缺失，一个是用来链接纵向的缺失。
kernelZ = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 30))
image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernelZ)

kernelY = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 30))
image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernelY)


#找到轮廓，并将轮廓在原图上画出来。
img,contours,hierarchy = cv2.findContours(image,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) #二值化后的数据，轮廓的种类(是否有父项等)，是否存储所有的轮廓点，
                                                                               
img2 = cv2.drawContours(img0, contours, -1, (0, 0, 255), 3)

#print (contours[0].shape)


#cv2.imwrite('two.png',img2)


# 用绿色(0, 255, 0)来画出最小的矩形框架 ,外接矩形，表示不考虑旋转并且能包含整个轮廓的矩形
#cnt = contours[0]   
#x, y, w, h = cv2.boundingRect(cnt)  #输入轮廓，返回4个值，左上角的坐标、宽度、高度
#cv2.rectangle(rectan, (x, y), (x+w, y+h), (0, 255, 0), 2)  #在img图像上画矩形。
#cv2.imwrite('rec1.png',rectan)


#最小外接矩，考虑了旋转
cnt = contours[0]
rect = cv2.minAreaRect(cnt)  # 最小外接矩形,返回值rect内包含该矩形的中心点坐标、高度宽度及倾斜角度等信息
#box_ = cv2.boc


rotate(rectan,abs(rect[2]))

box = np.int0(cv2.boxPoints(rect))  # 矩形的四个角点取整
#print (box)
cv2.drawContours(rectan, [box], 0, (255, 0, 0), 2)
cv2.imwrite('rec.png',rectan)


#print(rect[2])



'''


#recimage = rectan(box)
#cv2.imwrite('recimage.png',recimage)

#ret,image = cv2.threshold(bina,230,255,cv2.THRESH_BINARY_INV) #将灰度值大于230的都设置成白色，小于230的都设置成黑色。INV表示与之相反。



#kernelB = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (40,40)) #选择一个形态，这是个矩形，大小是3×3
#image = cv2.dilate(img1, kernelB)#膨胀，与腐蚀相反

#kernelA = cv2.getStructuringElement(cv2.MORPH_RECT, (40,40)) #选择一个形态，这是个矩形，大小是3×3
#image = cv2.dilate(image, kernelA)#膨胀，与腐蚀相反


#img,contours,hierarchy = cv2.findContours(image,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) #二值化后的数据，轮廓的种类(是否有父项等)，是否存储所有的轮廓点，

#img2 = cv2.drawContours(img0, contours, -1, (0, 0, 255), 3)


#输出的结果格式numpy ,list ,numpy
#二值化后的图，与thresh1一样，  列表形式存储的所有的轮廓坐标点、用4个量表示的轮廓属性(Next,Previos,First_child,Parent)

#print (image.shape,len(contours),hierarchy.shape)


#tmp, contours, w1 = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#img = cv2.drawContours(img1, contours, -1, (0, 0, 255), 3)


#cv2.imwrite('two.png',img)







#img0 = cv2.imread('new1.png') #read picture; default no alpha; but you can set 1 to include alpha.('new.png',1)
#                                #set 0 means read gray picture

#cut_img = img0[33:265,337:1454] #取出图像，注意仍然是先高度，再宽度，和其原始的shape是一样的。
#cv2.imwrite('test1.png',cut_img)

'''水平投影'''

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
            
    for i in range(len(V_End)):
        cut_img = origineImage[0:h,V_Start[i]:V_End[i]]        
        cv2.imwrite('{}.png'.format(i),cut_img)
    #cut_img = img0[33:265,337:1454] #取出图像，注意仍然是先高度，再宽度，和其原始的shape是一样的。   
    '''
    
    
      

    #以下程序将图片分割成上下2部分
    
    #读入原始图像
    origineImage = cv2.imread('3.png')
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
        cut_img = origineImage[H_Start[i]:H_End[i],0:w]        
        cv2.imwrite('{}.png'.format(i),cut_img)
    #cut_img = img0[33:265,337:1454] #取出图像，注意仍然是先高度，再宽度，和其原始的shape是一样的。 
    

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








