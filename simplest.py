#!/usr/bin/python
# -*- coding:UTF-8 -*-

import pycuda.autoinit
import pycuda.driver as drv
import numpy
 
from pycuda.compiler import SourceModule
mod = SourceModule("""
__global__ void multiply_them(float *dest, float *a, float *b,float *c)
{
  const int i = threadIdx.x;                            #threadIdx是与block紧密相关的。
  //const int i = blockIdx.x*blockDim.x + threadIdx.x;  #注意后面的block=(20,1,1), grid=(10,1))可以改成这种形式
                                                        #这个时候blockIdx是从0-9,blockDim==20,对结果无影响。
  dest[i] = a[i] * b[i];
  dest[i] = a[i] * b[i];
  c[i] = i;
}
""")
 
multiply_them = mod.get_function("multiply_them")
 
a = numpy.random.randn(200).astype(numpy.float32)     #生成符合标准正态分布的随机数组。并转换成单精度浮点数。
b = numpy.random.randn(200).astype(numpy.float32)
c    = numpy.zeros_like(a)                            #注意这里要定义一下dest，否则会提示dest未定义。
dest = numpy.zeros_like(a)
multiply_them(
        drv.Out(dest), drv.In(a), drv.In(b),drv.Out(c),   #drv主要是用来实现内存和显存之间的数据转移。
        block=(200,1,1), grid=(1,1))                      #这个400,1,1看起来是3维的，可能实际上仍然是1维的。
                                                          #可以将block固定，grid改成较大的数，对结果无影响。
print (c, dest-a*b )

