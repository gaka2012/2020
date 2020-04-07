#!/usr/bin/python
# -*- coding:UTF-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import random


def test(a,b):
    a+=2
    b+=2
    a0,b0 = [],[]
    a0.append(a)
    b0.append(b)
    return a0,b0

c=test(1,3)
print (c[0])
