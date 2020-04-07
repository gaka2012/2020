#!/usr/bin/python
# -*- coding:UTF-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import random
 
 
# (-1, 2)
# 初始化原始种群
def ori_popular(num):
    popular = []
    for i in range(num):
        x = random.uniform(-1, 2)  # 在此范围内生成一个随机浮点数
        popular.append(x)
    return popular
 
 
# 编码，也就是由表现型到基因型，性征到染色体,将实数转换为二进制的数。
def encode(popular):  # popular应该是float类型的列表
    popular_gene = []
    for i in range(0, len(popular)):               #公式 (b-a)/(Max-Min)*(Y-min)+a=x,其中a,b是[-1,2]Max是二进制最大值。x表示popular[i],下面的公式是求Y，即二进制数值。
        data = int((popular[i]-(-1)) / 3 * 2**18)  # 染色体序列为18bit，由于取值范围是[-1,2]，区间是3,如果保留4位有效数字，有3×10××4=3万个数。
        bin_data = bin(data)  # 整形转换成二进制是以字符串的形式存在的
        for j in range(len(bin_data)-2, 18):  # 序列长度不足补0
            bin_data = bin_data[0:2] + '0' + bin_data[2:]
        popular_gene.append(bin_data)
    return popular_gene
 
 
# 解码，即适应度函数。通过基因，即染色体得到个体的适应度值,返回的fitness实际上是100个计算后的Y值。
def decode(popular_gene):
    fitness = []
    for i in range(len(popular_gene)):
        x = (int(popular_gene[i], 2) / 2**18) * 3 - 1  #将基因装换为数值
        #value = x * np.sin(10 * np.pi * x) + 2        #函数公式
        value = x*x  
        fitness.append(value)
    return fitness
 
 
# 选择and交叉。选择用轮牌赌，交叉概率为0.66
def choice_ex(popular_gene):
    fitness = decode(popular_gene) #输入的参数是原始基因，需要先解码。适应度函数直接就是y值本身
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
        if is_change:  #如果上一行的代码的结果是1,2就交叉，是0就不交叉，所以概率是 2/3=0.66
            temp_s = temp[0][9:15]
            temp[0] = temp[0][0:9] + temp[1][9:15] + temp[0][15:]
            temp[1] = temp[1][0:9] + temp_s + temp[1][15:]     #基因总长度是18,交换其中的9-14,共计6个基因。
 
        popular_new.append(temp[0])
        popular_new.append(temp[1])
    return popular_new  #最后返回的是100个经过选择，交叉后的基因。由于选中最大值的概率更高，所以这100个基因中包含更多的高值基因。
 
 
# 变异.概率为0.05
def variation(popular_new): #输入的参数是经过选择交叉后的基因。
    for i in range(len(popular_new)): #100个基因
        is_variation = random.uniform(0, 1)
        # print([len(k) for k in popular_new])
        if is_variation < 0.02:   #变异，实际上就是对基因2-18中的某一个基因变成0或1
            rand = random.randint(2, 19)
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
    # 第一步：初始化原始种群, 一百个个体
    num = 100
    ori_popular = ori_popular(num) #返回一个列表，里面是[-1,2]区间内的100个随机数
    
    # 第二步：得到原始种群的基因，返回一个列表，里面是100个随机数对应的基因。
    ori_popular_gene = encode(ori_popular)  # 18位基因
    new_popular_gene = ori_popular_gene
    
    y = []
    for i in range(1000):  # 迭代次数。繁殖1000代
        new_popular_gene = choice_ex(new_popular_gene)  # 第三步：选择和交叉
        new_popular_gene = variation(new_popular_gene)  # 变异
        # 取当代所有个体适应度平均值
        new_fitness = decode(new_popular_gene)
        sum_new_fitness = 0
        for j in new_fitness:
            sum_new_fitness += j
        y.append(sum_new_fitness/len(new_fitness))
    
    # 画图 #横坐标是迭代次数，纵坐标是y值。
    x = np.linspace(0, 1000, 1000)
    fig = plt.figure(figsize=(25,15))  # 相当于一个画板
    axis = fig.add_subplot(111)  # 坐标轴
    axis.plot(x, y)
    plt.savefig('test')
    plt.show()
    
