#!/usr/bin/env python3
# coding: utf-8

import numpy as np

import argparse
from chainer import cuda, Variable

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')

parser.add_argument('--test', '-t', choices=['simple', 'mnist'],
                    default='mnist',
                    help='test type ("simple", "mnist")')


""" GPUで計算する時は、cupy = numpy + GPUで処理する。 """
args = parser.parse_args()
if args.gpu >= 0:
    cuda.check_cuda_available()
xp = cuda.cupy if args.gpu >= 0 else np

xp.seterr(all="ignore")

class SGD:

    """Stochastic Gradient Descent"""

    def __init__(self, lr=0.01, div_lr=10.):
        self.lr = lr
        self.div_lr = div_lr
        
    def update(self, params, grads, s_limit=1e-4):
        for key in params.keys():
            if key == "b":
                for (b, g) in zip(params[key], grads[key]):
                    b += self.lr * g
            if key == "W":
                for (w, g) in zip(params[key], grads[key]):
                    w += self.lr * g
            if key == "z":
                for (z, g) in zip(params[key], grads[key]):
                    z += (self.lr/self.div_lr) * g
            if key == "s":
                for i in range(len(params[key])):
                    params[key][i] *= 0.
                    params[key][i] += xp.maximum(xp.exp(params["z"][i]), s_limit)
            if key == "V":
                for(v, g) in zip(params[key], grads[key]):
                    v += self.lr * g

class Momentum:
    """Momentum SGD"""

    def __init__(self, lr=0.01, momentum=0.9, div_lr=10., lr_decay=0.):
        self.lr = lr
        self.div_lr = div_lr
        self.lr_decay = lr_decay
        self.momentum = momentum
        self.v = None
        self.iter = 0
        
    def update(self, params, grads, s_limit=1e-4):
        self.iter += 1
        lr = self.lr
        if self.lr_decay > 0:
            lr = lr * (1. / (1. + self.lr_decay * self.iter))

        if self.v is None:
            self.v = {}
            for key, vals in params.items():
                if key != "s":
                    v = []
                    for val in vals:
                        v.append(xp.zeros_like(val))
                
                    self.v[key] = v[:]
                
        for key in params.keys():
            if key == "z":
                for i in range(len(self.v[key])):
                    self.v[key][i] = self.momentum*self.v[key][i] + (lr/self.div_lr)*grads[key][i]
                    #self.v[key][i] = (lr/self.div_lr)*grads[key][i]
                    params[key][i] += self.v[key][i]
            elif key == "s":
                for i in range(len(params[key])):
                    params[key][i] *= 0.
                    params[key][i] += xp.maximum(xp.exp(params["z"][i]), s_limit)
            else:
                for i in range(len(self.v[key])):
                    if i == 0 or i == len(self.v[key])-1:
                        self.v[key][i] = self.momentum*self.v[key][i] + (lr/self.div_lr)*grads[key][i]
                    else:
                        self.v[key][i] = self.momentum*self.v[key][i] + lr*grads[key][i]
                    params[key][i] += self.v[key][i]

class Nesterov:

### not tested ###

    """Nesterov's Accelerated Gradient (http://arxiv.org/abs/1212.0901)"""

    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None
        
    def update(self, params, grads, s_limit=1e-4):
        if self.v is None:
            self.v = {}
            for key, vals in params.items():
                if key != "s":
                    v = []
                    for val in vals:
                        v.append(xp.zeros_like(val))
                
                    self.v[key] = v[:]
            
        for key in params.keys():
            if key == "s":
                for i in range(len(params[key])):
                    params[key][i] = xp.maximum(xp.exp(params["z"][i]), s_limit)
            else:
                for i in range(len(self.v[key])):
                    self.v[key][i] *= self.momentum
                    self.v[key][i] += self.lr * grads[key][i]
                    params[key][i] -= self.momentum * self.momentum * self.v[key][i]
                    params[key][i] += (1 + self.momentum) * self.lr * grads[key][i]

class AdaGrad:

    """AdaGrad"""

    def __init__(self, lr=0.01, epsilon=1e-8, lr_decay=0.):
        self.lr = lr
        self.h = None
        self.epsilon = epsilon
        self.iter = 0
        self.lr_decay = lr_decay
        
    def update(self, params, grads, s_limit=1e-4):
        self.iter += 1
        lr = self.lr
        if self.lr_decay > 0:
            lr = lr * (1. / (1. + self.lr_decay * self.iter))

        if self.h is None:
            self.h = {}
            for key, vals in params.items():
                if key != "s":
                    h = []
                    for val in vals:
                        h.append(xp.zeros_like(val))
                
                    self.h[key] = h[:]
            
        for key in params.keys():
            if key == "s":
                for i in range(len(params[key])):
                    params[key][i] = xp.maximum(xp.exp(params["z"][i]), s_limit)
            else:
                for i in range(len(self.h[key])):
                    self.h[key][i] += grads[key][i] * grads[key][i]
                    params[key][i] += self.lr * grads[key][i] / (xp.sqrt(self.h[key][i]) + self.epsilon )
            
class RMSprop:

    """RMSprop"""

    def __init__(self, lr=0.01, decay_rate = 0.9, epsilon=1e-8, lr_decay=0.):
        self.lr = lr
        self.decay_rate = decay_rate
        self.h = None
        self.epsilon = epsilon
        self.iter = 0
        self.lr_decay = lr_decay
        
    def update(self, params, grads, s_limit=1e-4):
        self.iter += 1
        lr = self.lr
        if self.lr_decay > 0:
            lr = lr * (1. / (1. + self.lr_decay * self.iter))

        if self.h is None:
            self.h = {}
            for key, vals in params.items():
                if key != "s":
                    h = []
                    for val in vals:
                        h.append(xp.zeros_like(val))
                
                    self.h[key] = h[:]
            
        for key in params.keys():
            if key == "s":
                for i in range(len(params[key])):
                    #params[key][i] *= 0
                    #params[key][i] += xp.maximum(xp.exp(params["z"][i]), s_limit)
                    params[key][i] = xp.maximum(xp.exp(params["z"][i]), s_limit)
            else:
                for i in range(len(self.h[key])):
                    self.h[key][i] *= self.decay_rate
                    self.h[key][i] += (1 - self.decay_rate) * grads[key][i] * grads[key][i]
                    params[key][i] += self.lr * grads[key][i] / (xp.sqrt(self.h[key][i]) + self.epsilon)
        
class Adam:

    """Adam (http://arxiv.org/abs/1412.6980v8)"""

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, div_lr=10., lr_decay=0., epsilon=1e-8, momentum=0.):
        self.lr = lr
        self.div_lr = div_lr
        self.lr_decay = lr_decay
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None
        self.vt = None
        self.epsilon = epsilon
        self.momentum = momentum
        
    def update(self, params, grads, s_limit=1e-4):
        if self.m is None:
            self.m, self.v, self.vt = {}, {}, {}
            for key, vals in params.items():
                if key != "s":
                    m, v, vt = [], [], []
                    for val in vals:
                        m.append(xp.zeros_like(val))
                        v.append(xp.zeros_like(val))
                        vt.append(xp.zeros_like(val))
                
                    self.m[key] = m[:]
                    self.v[key] = v[:]
                    self.vt[key] = vt[:]
        
        #self.iter += 1
        lr = self.lr
        if self.lr_decay > 0:
            lr = lr * (1. / (1. + self.lr_decay * self.iter))

        self.iter += 1
        lr_t  = lr * (xp.sqrt(1. - self.beta2**self.iter) / (1. - self.beta1**self.iter))

        #print(lr_t)
        
        for key in params.keys():
            if key == "s":
                for i in range(len(params[key])):
                    params[key][i] *= 0
                    params[key][i] += xp.maximum(xp.exp(params["z"][i]), s_limit)
                    #params[key][i] = xp.maximum(xp.exp(params["z"][i]), s_limit)
            else:
                for i in range(len(self.m[key])):
                    #self.m[key][i] += (1 - self.beta1) * (grads[key][i] - self.m[key][i])
                    #self.v[key][i] += (1 - self.beta2) * (grads[key][i] * grads[key][i] - self.v[key][i])

                    self.m[key][i] = (self.beta1 * self.m[key][i]) + (1. - self.beta1) * grads[key][i]
                    self.v[key][i] = (self.beta2 * self.v[key][i]) + (1. - self.beta2) * xp.square(grads[key][i])

                    if key == "z":
                        #self.vt[key][i] = self.momentum*self.vt[key][i] + lr_t*self.m[key][i] / (xp.sqrt(self.v[key][i]) + self.epsilon)
                        self.vt[key][i] = self.momentum*self.vt[key][i] + \
                                            (lr_t/self.div_lr)*self.m[key][i] / (xp.sqrt(self.v[key][i]) + self.epsilon)
                    else:
                        if i == 0 or i == len(self.m[key])-1:
                        #if i == 0:
                            self.vt[key][i] = self.momentum*self.vt[key][i] + \
                                                (lr_t/self.div_lr)*self.m[key][i] / (xp.sqrt(self.v[key][i]) + self.epsilon)
                        else:
                            self.vt[key][i] = self.momentum*self.vt[key][i] + \
                                                lr_t*self.m[key][i] / (xp.sqrt(self.v[key][i]) + self.epsilon)
                        
                    params[key][i] += self.vt[key][i]
                    #params[key][i] += lr_t * self.m[key][i] / (xp.sqrt(self.v[key][i]) + self.epsilon)
        
        """
        for key in params.keys():
            #self.m[key] = self.beta1*self.m[key] + (1-self.beta1)*grads[key]
            #self.v[key] = self.beta2*self.v[key] + (1-self.beta2)*(grads[key]**2)
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1 - self.beta2) * (grads[key]**2 - self.v[key])
            
            params[key] -= lr_t * self.m[key] / (xp.sqrt(self.v[key]) + 1e-7)
            
            #unbias_m += (1 - self.beta1) * (grads[key] - self.m[key]) # correct bias
            #unbisa_b += (1 - self.beta2) * (grads[key]*grads[key] - self.v[key]) # correct bias
            #params[key] += self.lr * unbias_m / (xp.sqrt(unbisa_b) + 1e-7)
        """
