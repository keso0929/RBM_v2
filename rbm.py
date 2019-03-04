#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
 Restricted Boltzmann Machine (RBM)
 References :
   - Y. Bengio, P. Lamblin, D. Popovici, and H. Larochelle: Greedy Layer-Wise
   Training of Deep Networks, Advances in Neural Information Processing
   Systems 19, 2007. 
"""

import sys, os
sys.path.append(os.pardir)

import argparse
import time

import numpy as np

from common.optimizer import *
from common.functions import sigmoid, softmax, softmax_act
from common.util import *

class RBM(object):
    def __init__(self, n_visible=2, n_hidden=3, \
        W=None, c=None, b=None, s=None, z=None, xp_rng=None, \
        mode="Bernoulli", sample_flag=1, pcd_flag=0, update_variance=False, \
        winit="normal", scale=0.001, n_categorical=0, n_c_dist=0, \
        optimizer="adam", optimizer_param={}, pretrain_visible=False):

        self.n_visible = n_visible  # num of units in visible (input) layer
        self.n_hidden = n_hidden    # num of units in hidden layer
        self.mode = mode
        self.sample_flag = sample_flag
        self.pcd_flag = pcd_flag
        self.prev_h = None

        # optimzer
        optimizer_class_dict = {"sgd":SGD, "momentum":Momentum, "nesterov":Nesterov,
                                "adagrad":AdaGrad, "rmsprop":RMSprop, 'adam':Adam}
        self.optimizer = optimizer_class_dict[optimizer.lower()](**optimizer_param)

        if xp_rng is None:
            xp_rng = np.random.RandomState(123)

        if W is None:
            if winit == "uniform":
                a = 1. / n_visible
                high = a
                low = -a
                #initial_W = np.array(xp_rng.uniform(
                #    low=-a,
                #    high=a,
                #    size=(n_visible, n_hidden)))
                #initial_W = (high - low) * xp_rng.rand(n_visible, n_hidden) + low
                initial_W = scale*(2 * xp_rng.rand(n_visible, n_hidden) + 1)
            elif winit == "normal":
                initial_W = scale * xp_rng.randn(n_visible, n_hidden)

            W = initial_W

        if c is None:
            c = np.zeros(n_hidden)  # initialize h bias 0

        if b is None:
            b = np.zeros(n_visible)  # initialize v bias 0

        if s is None:
            s = np.ones(n_visible)

        if z is None:
            z = np.zeros(n_visible)

        self.xp_rng = xp_rng
        self.W = np.array(W)
        self.c = np.array(c)
        self.b = np.array(b)
        self.z = np.array(z)
        self.s = np.array(s)

        self.update_variance = update_variance
        self.pretrain_visible = pretrain_visible

        if self.mode == "mix":
            self.n_categorical = n_categorical
            self.n_bernoulli = n_categorical
        
        self.params = {}
        self.params["W"] = [self.W]
        self.params["b"] = [self.b, self.c]

        if self.update_variance == True:
            self.params["z"] = [self.z]
            self.params["s"] = [self.s]

    def grad(self, v1, h1, v2, h2):
        if self.mode == "Bernoulli":
            dW = (np.dot(v1.T, h1) - np.dot(v2.T, h2)) / self.batchsize
            db = np.mean(v1 - v2, axis=0)
            dc = np.mean(h1 - h2, axis=0)
        elif self.mode == "Gaussian" or self.mode == "mix":
            vz1 = 0.5*np.square(v1 - self.b) - np.dot(h1, self.W.T)*v1
            vz2 = 0.5*np.square(v2 - self.b) - np.dot(h2, self.W.T)*v2

            dW =  (np.dot((v1/self.s).T, h1) - np.dot((v2/self.s).T, h2)) / self.batchsize
            db = np.mean(v1/self.s - v2/self.s, axis=0)
            dc = np.mean(h1 - h2, axis=0)
            
            if self.update_variance == True:
                dz = np.mean(vz1/self.s - vz2/self.s, axis=0)

                if self.mode == "mix":
                    _, dz_g = np.hsplit(dz, [self.n_categorical])
                    dz = np.concatenate([np.zeros(self.n_categorical), dz_g], axis=0)

        grads = {}
        grads["W"] = [dW]
        grads["b"] = [db, dc]
        if self.update_variance:
            grads["z"] = [dz]
            #self.z += self.lr * dz
            #self.s = np.maximum(np.exp(self.z), 0.001)

        return grads

    def train(self, input, n_epoch=50, batchsize=10, k=1, lr=0.1, \
            cost="reconstruction error", momentum=0.0, decay=0.0, div_lr=1., lr_decay=0.):
        self.k = k
        #self.momentum = momentum
        #self.decay = decay
        self.batchsize = batchsize
        self.lr = lr
        
        self.optimizer.div_lr = div_lr
        self.optimizer.lr = lr
        self.optimizer.lr_decay = lr_decay
        self.optimizer.momentum = momentum

        V_train = input
        N = V_train.shape[0]
        loss = []

        if cost == "free energy":
            get_cost = self.get_cost
        elif cost == "reconstruction cross entropy":
            get_cost = self.get_reconstruction_cross_entropy
        elif cost == "reconstruction error":
            get_cost = self.get_reconstruction_error
        
        for epoch in range(0, n_epoch):
            perm = np.random.permutation(N)
            l = 0.
            for i in range(0, N, batchsize):
                v_batch = np.asarray(V_train[perm[i:i+batchsize]])
                v1 = v_batch
                h1, v2, h2 = self.contrastive_divergence(v1, self.k)

                grads = self.grad(v1, h1, v2, h2)  ###
                self.optimizer.update(self.params, grads)
                
                l += np.sum(np.square(v1 - v2)) / self.batchsize

            loss.append(l*self.batchsize/N)

        return loss
        
    def contrastive_divergence(self, v, k=1):
        v1 = v

        if self.prev_h is None or v1.shape[0] != self.batchsize:
            h1, hs = self.sample_h_given_v(v1)
        else:
            h1 = self.prev_h
            hs = self.xp_rng.binomial(size=h1.shape, n=1, p=h1)

        chain_start = hs

        for step in range(k):
            if step == 0:
                v2, vs = self.sample_v_given_h(chain_start)
                h2, hs = self.sample_h_given_v(vs)
            else:
                v2, vs = self.sample_v_given_h(hs)
                h2, hs = self.sample_h_given_v(vs)

        v2 = np.copy(vs)

        if self.pcd_flag:
            if h2.shape[0] == self.batchsize:
                self.prev_h = h2
            else:
                self.prev_h = None

        return h1, v2, h2

    def sample_h_given_v(self, v):
        h = self.propup(v)
        if self.sample_flag == 0:
            return h, h
        
        hs = self.xp_rng.binomial(size=h.shape, n=1, p=h)
        
        return h, hs

    def sample_v_given_h(self, h):
        v = self.propdown(h)
        if self.sample_flag == 0:
            return v, v
        
        if self.mode =="Bernoulli":
            vs = self.xp_rng.binomial(size=v.shape, n=1, p=v)
        elif self.mode == "Gaussian":
            vs = (self.xp_rng.randn(v.shape[0], v.shape[1]) * np.sqrt(self.s)) + v
        elif self.mode == "mix":
            v_b, v_g = np.hsplit(v, [self.n_categorical])
            vs_g = (self.xp_rng.randn(v_g.shape[0], v_g.shape[1]) * np.sqrt(self.s[self.n_categorical:])) + v_g
            vs_b = softmax_act(v_b, self.n_categorical, sample=1)
            vs = np.concatenate([vs_b, vs_g], axis=1)

        return v, vs

    def propup(self, v):
        if self.mode == "Bernoulli":
            pre_sigmoid_activation = np.dot(v, self.W) + self.c
        elif self.mode == "Gaussian" or self.mode == "mix":
            if self.pretrain_visible == False:
                pre_sigmoid_activation = np.dot(v/self.s, self.W) + self.c
            else:
                pre_sigmoid_activation = 2*(np.dot(v/self.s, self.W) + self.c)

        return sigmoid(pre_sigmoid_activation)

    def propdown(self, h):
        if self.mode == "Bernoulli":
            v_mean = sigmoid(np.dot(h, self.W.T) + self.b)
        elif self.mode == "Gaussian":
            v_mean = np.dot(h, self.W.T) + self.b
        elif self.mode == "mix":
            pre_sigmoid_activation = np.dot(h, self.W.T) + self.b
            pre_b, pre_g = np.hsplit(pre_sigmoid_activation, [self.n_categorical])
            v_mean_g = pre_g
            
            pre_b = np.hsplit(pre_b, [42, 2*42])
            ret = []
            v_mean_b = []
            for pre_bi in pre_b:
                ret.append(softmax(pre_bi))
            for reti in ret:
                if v_mean_b == []:
                    v_mean_b = reti
                else:
                    v_mean_b = np.concatenate([v_mean_b, reti], axis=1)  
                        
            v_mean = np.concatenate([v_mean_b, v_mean_g], axis=1)
            
        return v_mean

    def get_reconstruction_cross_entropy(self,v_test):
        pre_sigmoid_activation_h = np.dot(v_test, self.W) + self.c
        sigmoid_activation_h = sigmoid(pre_sigmoid_activation_h)

        pre_sigmoid_activation_v = np.dot(sigmoid_activation_h, self.W.T) + self.b
        sigmoid_activation_v = sigmoid(pre_sigmoid_activation_v)

        cross_entropy =  - np.mean(
            np.sum(v_test * np.log(sigmoid_activation_v) +
            (1 - v_test) * np.log(1 - sigmoid_activation_v),
                      axis=1))

        return cross_entropy

    def get_reconstruction_error(self, v_test):
        reconstructed_v = self.reconstruct(v_test)

        return np.sum(np.square(np.abs(v_test - reconstructed_v))) / self.batchsize

    def reconstruct(self, v_in):
        v = np.copy(v_in, k=10)

        for i in range(k):
            h = rbm.propup(v)
            v = rbm.propdown(h)
            
        return v

    def get_cost(self, v_test):
        k = self.k
        
        _, vh_test, nh_test = self.contrastive_divergence(v_test, k)
        loss = self.free_energy(v_test) - self.free_energy(vh_test)
        
        return loss

    def free_energy(self, v_test):
        wx_b = np.dot(v_test, self.W) + self.c
        if self.mode == "Bernoulli":
            b_term = np.sum(np.dot(v_test, self.b.T))
        elif self.mode == "Gaussian":
            v_ = v_test - self.b
            b_term = np.sum(0.5 * v_ * v_)

        hidden_term = np.sum(np.log(1 + np.exp(wx_b)))

        return -hidden_term - b_term

if __name__ == "__main__":
    pass
