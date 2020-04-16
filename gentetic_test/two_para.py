#!/usr/bin/python
# -*- coding:UTF-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import random
 
 
# (-1, 2)
# 初始化原始种群,输入种群的数量以及最大值，最小值。
def num_popular(num,max_num,min_num):
    popular = []
    for i in range(num):
        x = random.uniform(min_num, max_num)  # 在此范围内生成一个随机浮点数
        x = round(x,2)
        popular.append(x)
    return popular
 
 
# 编码，也就是由表现型到基因型，性征到染色体,将实数转换为二进制的数。
def encode(gene_dict):
    new_gene = {} #输入的是一个字典，键值是每一初始种群的最大、最小、以及基因长度，对应的值是其初始化后的n个x值，返回的只是把x值转换为了基因。
    for key,value in gene_dict.items():
        num_range = key[0]-key[1]
        tem_ge = []
        for i in range(0,len(value)): #把每一个x值转换为基因
            data = int((value[i]-(key[1])) / num_range * 2**key[2])  # 染色体序列为18bit，由于取值范围是[-1,2]，区间是3,如果保留4位有效数字，有3×10××4=3万个数。
            bin_data = bin(data)
            for j in range(len(bin_data)-2, key[2]):  # 序列长度不足补0
                bin_data = bin_data[0:2] + '0' + bin_data[2:]
            tem_ge.append(bin_data) 
        new_gene[key] = tem_ge
    return new_gene   
'''
def encode(popular,max_num,min_num,gene_len):  # popular应该是float类型的列表
    popular_gene = []
    num_range    = max_num-min_num
    for i in range(0, len(popular)):               #公式 (b-a)/(Max-Min)*(Y)+a=x,其中a,b是[-1,2],Max是二进制最大值。x表示popular[i],下面的公式是求Y，即二进制数值。
        data = int((popular[i]-(min_num)) / num_range * 2**gene_len)  # 染色体序列为18bit，由于取值范围是[-1,2]，区间是3,如果保留4位有效数字，有3×10××4=3万个数。
        bin_data = bin(data)  # 整形转换成二进制是以字符串的形式存在的
        for j in range(len(bin_data)-2, gene_len):  # 序列长度不足补0
            bin_data = bin_data[0:2] + '0' + bin_data[2:]
        popular_gene.append(bin_data)
    return popular_gene
'''
 
# 解码，即适应度函数。通过基因，即染色体得到个体的适应度值,返回的fitness实际上是100个计算后的Y值。
def decode(gene_dict):
    x_list = [] #存储基因翻译过来的数值。列表中有列表，有几个参数，就有几个列表。
    fitness= [] #存储适应度函数的结果，即n个y值。
    #输入的参数是一个字典，键是是每一初始种群的最大、最小、以及基因长度，对应的值是其初始化后的n个x值对应的基因。
    for key,value in gene_dict.items():
        num_range = key[0]-key[1]
        tem_list  = [] #存储每个参数对应的x值。
        for ge in range(len(value)):
            x = (int(value[ge],2) / 2**key[2]) * num_range - key[1]  #将基因装换为数值
            tem_list.append(x)
        x_list.append(tem_list)
    
    #对输入的多个参数对应的x值求适应度函数
    for i in range(len(x_list[0])): #每个参数n个基因，数量是相同的。
        y_value = x_list[0][i]**2+x_list[1][i]*3 #函数y=x**2+3z
        fitness.append(y_value)  
    return fitness,x_list
      
'''
def decode(popular_gene,max_num,min_num,gene_len):
    fitness = []
    num_range = max_num-min_num
    new_num = []  #存储将基因转换为数值后的数。
    for i in range(len(popular_gene)):
        x = (int(popular_gene[i], 2) / 2**gene_len) * num_range - min_num  #将基因装换为数值
        #value = x * np.sin(10 * np.pi * x) + 2        #函数公式
        new_num.append(x)
        value = x*x  
        fitness.append(value)
    return fitness,new_num
    #return new_num #返回一个列表，存储将基因装换为数值后的列表。
'''
def choice_ex(gene_dict):
    
    #将所有的初始基因放在一个列表下，方便后面的基因交换，节省时间。
    value_list = []
    para_num   = 0 #看一下有几个参数
    for key,value in gene_dict:
        value_list.append(value)
        para_num+=1
        gene_len = key[2]    
    #求适应度函数的和
    fitness = decode(gene_dict)[0]#输入的参数是原始基因字典，需要先解码。返回值是适应度函数y值本身
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
            
    #选择
    tem_popular_new = [] #列表，存储基因交换后的基因
    for i in range(para_num):
        te = []
        tem_popular_new.append(te)
        
    for i in range(int(len(fitness)/2)): #一共有100个原始数据，将其分成50组，每组2个原始数据，2个基因，然后生成2个随机数字(范围是0-1)
                                         #看一下这2个数字的范围符合哪2个数据的概率分布，从而把这2个基因挑出来。
        temp = [] #存储临时选出来的基因，有几个参数就选几个。
        for n in range(para_num):
            te = []
            temp.append(te)
            
        for j in range(2):
            rand = random.uniform(0, 1)  # 在0-1之间随机一个浮点数
            for k in range(len(fitness)):#每一个数字都有一个累计概率，选出rand中的数字与哪个x值的累计概率项符合。
                if k == 0:
                    if rand < probability_sum[k]:
                        for m in range(para_num): #有m个参数，所以要把m个参数对应的基因都挑出来。
                            temp[m].append(value_list[m][k])
                else:
                    if (rand > probability_sum[k-1]) and (rand < probability_sum[k]):
                        for m in range(para_num):
                            temp[m].append(value_list[m][k])
                            
        # 交叉，交叉率为0.66。上面的temp会生成2个基因。
        is_change = random.randint(0, 2) #随机生成0-2范围内的整数。
        change_len=int(gene_len/2)  #交换的基因的长度，如果是4,则交换其中的2个基因。注意基因的前2个是符号0b，不参与交换。
        if is_change:  #如果上一行的代码的结果是1,2就交叉，是0就不交叉，所以概率是 2/3=0.66
            for h in range(len(temp)):
                temp_s = temp[h][0][1+change_len:1+change_len*2]
                temp[h][0] = temp[h][0][0:1+change_len] + temp[h][1][1+change_len:1+change_len*2] + temp[h][0][1+change_len*2:]
                temp[h][1] = temp[h][1][0:1+change_len] + temp_s + temp[h][1][1+change_len*2:] #基因总长度是18,交换其中的9-14,共计6个基
        
        for a in range(para_num):
            tem_popular_new[a]
        tem_popular_new
        popular_new.append(temp[0])
        popular_new.append(temp[1])

'''
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
    for i in range(int(len(fitness)/2)): #一共有100个原始数据，将其分成50组，每组2个原始数据，2个基因，然后生成2个随机数字(范围是0-1)
                                         #看一下这2个数字的范围符合哪2个数据的概率分布，从而把这2个基因挑出来。
        temp = []
        for j in range(2):
            rand = random.uniform(0, 1)  # 在0-1之间随机一个浮点数
            for k in range(len(fitness)):#每一个数字都有一个累计概率，选出rand中的数字与哪个x值的累计概率项符合。
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
'''
 
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
    # 第一步：初始化原始种群, 多个参数，每个参数都有4个变量，个体数目，取值的最大值，最小值,基因的长度。初始长度要保持一致，最大最小值不能一样。
    multi_para = [[10,10,0,8],[10,15,0,8]]
    
    #返回的原始种群ori_popular是一个字典，里面的键值是每一初始种群的最大、最小、以及基因长度，对应的值是其初始化后的n个x值。
    ori_popular = {}
    for i in multi_para:
        tem_popular = num_popular(i[0],i[1],i[2])
        ori_popular[i[1],i[2],i[3]] = tem_popular
    
    
'''
    #测试，输出原始分配的2个参数的值范围，以及值，
    for key,value in ori_popular.items():
        print (key)
        print (value)
    #测试，输入得到的适应度函数，以及基因转换后的x值。
    ori_popular = encode(ori_popular)
    test_fit = decode(ori_popular)[0]
    test_x   = decode(ori_popular)[1]
    print(test_fit)
    for i in test_x:
        print (i)  
'''        

    # 第二步：得到原始种群的基因，返回一个列表，里面是100个随机数对应的基因。
    ori_popular_gene = encode(ori_popular)  #返回的ori_popular_gene是一个字典，键是[max,min,基因长度]，对应的值是n个基因
    new_popular_gene = ori_popular_gene
    #print (new_popular_gene)
    #result = decode(new_popular_gene,max_num,min_num)
    #print (max(result))
    

    y = []
    for i in range(1000):  # 迭代次数。繁殖1000代
        new_popular_gene = choice_ex(new_popular_gene,max_num,min_num,gene_len)  # 第三步：选择和交叉
        new_popular_gene = variation(new_popular_gene,gene_len)  # 变异
        # new_fitness是一个列表，存储每个x值对应的y值，对这些y值求和，然后处以y值的数量，得到平均y值。
        new_fitness = decode(new_popular_gene,max_num,min_num,gene_len)[0]
        
        #每次迭代后剩下的x值会越来越向着最佳x值逼近，new_x存储每次迭代后的x值列表。
        new_x = decode(new_popular_gene,max_num,min_num,gene_len)[1]
        #查看第i次迭代后的x值列表
        if i ==5:
            print (new_x)
        
        #求每次迭代后的y值的平均值。
        sum_new_fitness = 0
        for j in new_fitness:
            sum_new_fitness += j
        y.append(sum_new_fitness/len(new_fitness)) #每次迭代都能得到一个平均y值，
    
    # 画图 #横坐标是迭代次数，纵坐标是y值。
    x = np.linspace(0, 1000, 1000)
    fig = plt.figure(figsize=(25,15))  # 相当于一个画板
    axis = fig.add_subplot(111)  # 坐标轴
    axis.plot(x, y)
    plt.savefig('test')
    #plt.show()
    plt.close()
    
'''
    
