#!/usr/bin/env python3
# coding: utf-8

import numpy as np
#from util import *

import argparse
from chainer import cuda, Variable

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')

parser.add_argument('--test', '-t', choices=['simple', 'mnist'],
                    default='mnist',
                    help='test type ("simple", "mnist")')

args = parser.parse_args()
if args.gpu >= 0:
    cuda.check_cuda_available()
xp = cuda.cupy if args.gpu >= 0 else np

xp.seterr(all="ignore")

def identity_function(x):
    return x

def step_function(x):
    return xp.array(x > 0, dtype=xp.int)

def sigmoid(x, a=1.):
    return 1. / (a*(1. + xp.exp(-x)))   

def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)

def relu(x):
    return xp.maximum(0, x)

def relu_grad(x):
    grad = xp.zeros(x)
    grad[x>=0] = 1
    return grad  

def softmax(x):
    y = xp.exp(x)
    y = (y.T / xp.sum(y.T, axis=0)).T

    return y

def softmax_act(x, n=42, sample=False):
    x = xp.hsplit(xp.copy(x), [42, 2*42])
    tmp = []
    ret = []
    for xi in x:
        tmp.append(softmax(xi))

    for tmpi in tmp:
        if sample == True:
            index = xp.argmax(tmpi, axis=1)
            zero = xp.zeros_like(tmpi)
            for i in range(len(index)):
                zero[i][index[i]] = 1.
            tmpi = zero
        if ret == []:
            ret = tmpi
        else:
            ret = xp.concatenate([ret, tmpi], axis=1)

    return ret

def softmax_onehot(x, n=42):
    tmp = xp.hsplit(x, [42, 2*42])
    ret = []

    for tmpi in tmp:
        index = xp.argmax(tmpi, axis=1)
        zero = xp.zeros_like(tmpi)
        for i in range(len(index)):
            zero[i][index[i]] = 1.
        tmpi = zero
        if ret == []:
            ret = tmpi
        else:
            ret = xp.concatenate([ret, tmpi], axis=1)

    return ret

def mean_squared_error(y, t):
    return 0.5 * xp.sum((y-t)**2)

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    # 教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
    if t.size == y.size:
        t = t.argmax(axis=1)
             
    batch_size = y.shape[0]
    return -xp.sum(xp.log(y[xp.arange(batch_size), t])) / batch_size

def softmax_loss(X, t):
    y = softmax(X)
    return cross_entropy_error(y, t)


"""
x = xp.random.rand(10, 42*3)
print(x.shape)

print(xp.argmax(x[0][:42]))
print(xp.argmax(x[0][42:42*2]))
print(xp.argmax(x[0][42*2:42*3]))

x = softmax_onehot(x)
print(xp.argmax(x[0][:42]))
print(xp.argmax(x[0][42:42*2]))
print(xp.argmax(x[0][42*2:42*3]))
"""
