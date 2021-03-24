import numpy as np
import random
import math
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import os
import pandas as pd
import pickle
import xlwt
from xlwt import Workbook 
from xlrd import open_workbook
import time



def tanh(x):
    t=(np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
    dt=1-t**2
    return t,dt

def sigmoid(x):
    s=1/(1+np.exp(-x))
    ds=s*(1-s)  
    return s,ds

def relu(x):
    if x > 0:
        r = x
        dr = 1
    else:
        r = 0
        dr = 0
    return r,dr

def lrelu(x):
    if x > 0:
        r = x
        dr = 1
    else:
        r = 0.1*x
        dr = 0.1
    return r,dr


def swish(x):
    sw = x/(1+np.exp(-x))
    dsw = sw + (1 - sw)
    return sw,dsw


dim = 1000

y_sigmoid = []
y_tanh = []
y_relu = []
y_swish = []
y_lrelu = []

x = np.arange(1, dim + 1)*6/dim - 3

for i in x:
    y_sigmoid.append(sigmoid(i)[0])
    y_tanh.append(tanh(i)[0])
    y_relu.append(relu(i)[0])
    y_swish.append(swish(i)[0])
    y_lrelu.append(lrelu(i)[0])


plt.figure()
plt.plot(x, y_sigmoid)   
plt.title("sigmoid")
plt.savefig('sigmoid.png')
plt.close()
plt.figure()
plt.plot(x, y_tanh)   
plt.title("tanh")
plt.savefig('tanh.png')
plt.close()
plt.figure()
plt.plot(x, y_relu)   
plt.title("relu")
plt.savefig('relu.png')
plt.close()
plt.figure()
plt.plot(x, y_swish)   
plt.title("swish")
plt.savefig('swish.png')
plt.close()
plt.figure()
plt.plot(x, y_lrelu)   
plt.title("lrelu")
plt.savefig('lrelu.png')
plt.close()