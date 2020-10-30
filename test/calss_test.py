#!/usr/bin/python
# -*- coding:UTF-8 -*-




class Person(object):                  #object:  基础类，超类，所有类的父类，没有合适的父类都可以写object,可以不写object
    def __init__ (self,name,age,sex='she'):  #构造函数，传递参数,默认参数可以在调用时重新定义，否则就是默认
        self.name1 = name              #类中的参数
        self.age  = age
        self.sex  = sex
        self.grade= 10                 #不占用传递参数
        self.build(sex='she')            #也可以将一个子函数当成默认参数使用!!!!会直接调用。

    def author1(self):
        a=2
        b=3
        c=a+b
        self.c = c
        #return c
        print (self.c)
    def author2(self):
        a=2
        d=a +a 
        self.d = d
        return d    

    def build(self,sex='she'):
        if sex in ['she']:
            self.author1()   #直接调用这个子函数。
            print ('bulid is running')
        return 0
