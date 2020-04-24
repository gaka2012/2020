#!/usr/bin/python
# -*- coding:UTF-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import random


#格雷编码返回到二进制编码,输入一个格雷编码基因，返回一个二进制基因。
def gray_decode(bin_num):
    te_str = ''
    an_point = int(bin_num[2])  #初始锚点是左边第一个基因。
    for i in range(3,len(bin_num)):
        te = an_point^int(bin_num[i]) #每次异或完后的基因作为新的锚点
        an_point = te
        te_str+= str(te)
    return bin_num[0:3]+te_str

#二进制吗到格雷编码，输入一个二进制基因，返回一个格雷编码。
def gray_encode(bin_num):
    te_str = ''
    for i in range(len(bin_num)-1,2,-1):
        te = int(bin_num[i])^int(bin_num[i-1])
        te_str=''.join([str(te),te_str])
    return bin_num[0:3]+ te_str         
 
# (-1, 2)
def ori_popular(num,max_num,min_num):
    popular = []
    te = []
    for i in range(min_num,max_num+1):
        te.append(i)
    j=0
    #根据num的值，增加popular的数量。
    for i in range(num):
        if j>(len(te)-1):
            j=0
            x = te[j]
        else :
            x = te[j] 
            x = round(x,2)
        popular.append(x)
        j+=1
    #random.shuffle(popular) #将顺序打乱，貌似能稍微好点。
    return popular
'''
# 初始化原始种群,输入种群的数量以及最大值，最小值。
def ori_popular(num,max_num,min_num):
    popular = []
    for i in range(num):
        #x = random.uniform(min_num, max_num)  # 在此范围内生成一个随机浮点数
        #x = round(x,2)
        x = random.randint(min_num,max_num)
        popular.append(x)
    return popular
'''
 
# 编码，也就是由表现型到基因型，性征到染色体,将实数转换为二进制的数。
def encode(popular,max_num,min_num,gene_len):  # popular应该是float类型的列表
    popular_gene = []
    num_range    = max_num-min_num
    for i in range(0, len(popular)):               #公式 (b-a)/(Max-Min)*(Y)+a=x,其中a,b是[-1,2],Max是二进制最大值。x表示popular[i],下面的公式是求Y，即二进制数值。
        data = int((popular[i]-(min_num)) / num_range * (2**gene_len-1))  # 染色体序列为18bit，由于取值范围是[-1,2]，区间是3,如果保留4位有效数字，有3×10××4=3万个数。
        bin_data = bin(data)  # 整形转换成二进制是以字符串的形式存在的
        for j in range(len(bin_data)-2, gene_len):  # 序列长度不足补0
            bin_data = bin_data[0:2] + '0' + bin_data[2:]
        popular_gene.append(gray_encode(bin_data))
        #popular_gene.append(bin_data)
    return popular_gene
 
 
# 解码，即适应度函数。通过基因，即染色体得到个体的适应度值,返回的fitness实际上是100个计算后的Y值。
def decode(popular_gene,max_num,min_num,gene_len):
    fitness = []
    num_range = max_num-min_num
    new_num = []  #存储将基因转换为数值后的数。
    for i in range(len(popular_gene)):
        temp_ge = gray_decode(popular_gene[i])
        x = (int(temp_ge,2) / (2**gene_len-1)) * num_range + min_num  #将基因装换为数值
        #value = x * np.sin(10 * np.pi * x) + 2        #函数公式
        new_num.append(x)
        value = x*x  
        fitness.append(value)
    return fitness,new_num
    #return new_num #返回一个列表，存储将基因装换为数值后的列表。
 
# 选择and交叉。选择用轮牌赌，交叉概率为0.66
def choice_ex(popular_gene,max_num,min_num,gene_len):
    fitness = decode(popular_gene,max_num,min_num,gene_len)[0] #输入的参数是原始基因，需要先解码。适应度函数直接就是y值本身
    sum_fit_value = 0              #所有y值的和，由于是求最大值，适应度函数直接就是y值本身。所以这个实际上求的是所有适应度值的和。
    for i in range(len(fitness)):
        sum_fit_value += fitness[i]
    # 各个个体被选择的概率
    probability = []
    for i in range(len(fitness)):  #每个个体被选择的概率是其本身的适应度除以总的适应度
        probability.append(fitness[i]/sum_fit_value)
    # 概率分布
    probability_sum = []          #这个列表存储的是概率的和，最终的概率和是1.相当于染色体的累计概率
    for i in range(len(fitness)):
        if i == 0:
            probability_sum.append(probability[i])
        else:
            probability_sum.append(probability_sum[i-1] + probability[i])


    
    # 选择    
    popular_new = []
    for i in range(int(len(fitness)/2)): #一共有100个原始数据，将其分成50组，每组2个基因，然后生成2个随机数字(范围是0-1)
                                         #看一下这2个数字的范围符合哪个基因的概率分布，从而把这2个基因挑出来。
        temp = []
        for j in range(2):
            rand = random.uniform(0, 1)  # 在0-1之间随机一个浮点数
            for k in range(len(fitness)):
                if k == 0:
                    if rand < probability_sum[k]:
                        temp.append(popular_gene[k])
                else:
                    if (rand > probability_sum[k-1]) and (rand < probability_sum[k]):
                        temp.append(popular_gene[k])
 
        # 交叉，交叉率为0.66。上面的temp会生成2个基因。
        is_change = random.randint(0, 2) #随机生成0-2范围内的整数。
        change_len=int(gene_len/2)  #交换的基因的长度，如果是4,则交换其中的2个基因。注意基因的前2个是符号0b，不参与交换。
        if is_change:  #如果上一行的代码的结果是1,2就交叉，是0就不交叉，所以概率是 2/3=0.66
            temp_s = temp[0][1+change_len:1+change_len*2]
            temp[0] = temp[0][0:1+change_len] + temp[1][1+change_len:1+change_len*2] + temp[0][1+change_len*2:]
            temp[1] = temp[1][0:1+change_len] + temp_s + temp[1][1+change_len*2:] #基因总长度是18,交换其中的9-14,共计6个基
 
        popular_new.append(temp[0])
        popular_new.append(temp[1])
    return popular_new  #最后返回的是100个经过选择，交叉后的基因。由于选中最大值的概率更高，所以这100个基因中包含更多的高值基因。
 
 
# 变异.概率为0.05
def variation(popular_new,gene_len): #输入的参数是经过选择交叉后的基因。
    for i in range(len(popular_new)): #100个基因
        is_variation = random.uniform(0, 1)
        # print([len(k) for k in popular_new])
        if is_variation < 0.02:   #变异，实际上就是对基因2-18中的某一个基因变成0或1
            rand = random.randint(2, gene_len+1)
            if popular_new[i][rand] == '0':
                popular_new[i] = popular_new[i][0:rand] + '1' + popular_new[i][rand+1:]
            else:
                popular_new[i] = popular_new[i][0:rand] + '0' + popular_new[i][rand+1:]
    return popular_new
 
#需要修改的参数：
#(1)num=100; 初始化的取值数量，比如求[-1,2]区间y=x**2的最大值，随机取100个x值;注意在第一步的子函数中修改取值范围。
#(2)第二步，注意修改子函数中的基因长度， 
#(3)第三步，首先要解码，注意更改解码公式，以及函数公式。
if __name__ == '__main__':  # alt+enter
    # 第一步：初始化原始种群, 一百个个体,取值范围最大最小值。
    num,max_num,min_num = 100,63,0
    gene_len = 6 
    ori_popular = ori_popular(num,max_num,min_num) #返回一个列表，里面是[-1,2]区间内的100个随机数
    
    #print (ori_popular)
    # 第二步：得到原始种群的基因，返回一个列表，里面是100个随机数对应的基因。
    ori_popular_gene = encode(ori_popular,max_num,min_num,gene_len)  # 18位基因
    
    #输出测试
    #new_test = decode(ori_popular_gene,max_num,min_num,gene_len)[1]
    #gray_list = [gray_decode(i) for  i in ori_popular_gene]
    
    new_popular_gene = ori_popular_gene
    #print (new_popular_gene)
    #result = decode(new_popular_gene,max_num,min_num)
    #print (max(result))


    all_x = [] #存储所有的x值。
    y = []
    for i in range(1000):  # 迭代次数。繁殖1000代
        new_popular_gene = choice_ex(new_popular_gene,max_num,min_num,gene_len)  # 第三步：选择和交叉
        #new_popular_gene = variation(new_popular_gene,gene_len)  # 变异
        # new_fitness是一个列表，存储每个x值对应的y值，对这些y值求和，然后处以y值的数量，得到平均y值。
        new_fitness = decode(new_popular_gene,max_num,min_num,gene_len)[0]
        
        #每次迭代后剩下的x值会越来越向着最佳x值逼近，new_x存储每次迭代后的x值列表。
        new_x = decode(new_popular_gene,max_num,min_num,gene_len)[1]
        all_x.append(new_x)
        #print (new_x)
        #查看第i次迭代后的x值列表
        #if i ==5:
        #    print (new_x)
        
        #求每次迭代后的y值的平均值。
        sum_new_fitness = 0
        for j in new_fitness:
            sum_new_fitness += j
        y.append(sum_new_fitness/len(new_fitness)) #每次迭代都能得到一个平均y值，



    #找到最大的目标函数值，看看有几个。
    max_y = max(y)
    m = 0
    for j in range(len(y)):
        if y[j]==max_y:
            m+=1
            fc= open('max_x.txt','a+')
            fc.write(str(all_x[j]))
            fc.write('\n')
            fc.close()    
    print('there are %s max_y and it is %s'%(m,max_y))

    # 画图 #横坐标是迭代次数，纵坐标是y值。
    x = np.linspace(0, 1000, 1000)
    fig = plt.figure(figsize=(25,15))  # 相当于一个画板
    axis = fig.add_subplot(111)  # 坐标轴
    axis.plot(x, y)
    plt.savefig('one_gray.png')
    #plt.show()
    plt.close()

    
    
