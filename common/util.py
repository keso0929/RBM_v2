#!/usr/bin/env python3
# coding: utf-8

import numpy as np
import os

def softmax(x):
    y = np.exp(x)
    y = (y.T / np.sum(y.T, axis=0)).T

    return y

def split(data, index=5, start=0, end=-1):
    return data[:,index][start:end]

def plot_error(loss, loss_func="MSE", xlabel="epoch", loc=1, title=None, loss_test=None):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8,6))
    plt.plot(range(len(loss)), loss, lw=2, color="b", label="train")

    if not title is None:
        plt.title(title)

    if not loss_test is None:
        plt.plot(range(len(loss_test)), loss_test, lw=2, color="r", label="test")
        
    plt.legend(loc=loc)
    plt.xlabel(xlabel)
    plt.ylabel(loss_func)
    plt.show()

def load_mnist(test_size=0.3, binary_label=True, with_continuous_label=False):
    from sklearn.datasets import fetch_mldata
    from sklearn.preprocessing import LabelBinarizer
    from sklearn.cross_validation import train_test_split
    
    mnist = fetch_mldata("MNIST original", data_home=".")
    X = mnist.data
    y = mnist.target

    X = X.astype(np.float64)
    X /= X.max()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    label = y_train
    
    if binary_label == True:
        y_train = LabelBinarizer().fit_transform(y_train)
        y_test = LabelBinarizer().fit_transform(y_test)

    if with_continuous_label == False:
        return X_train, X_test, y_train, y_test
    else:
        return X_train, X_test, y_train, y_test, label

def plot_mnist(datas, labels, perm=None):
    import pylab
    
    if perm is None:
        p = np.random.random_integers(0, len(datas), 25)
    else:
        p = perm[:25]
        
    for index, i in enumerate(p):
        data = datas[i]
        label = labels[i]
        pylab.subplot(5, 5, index + 1)
        pylab.axis("off")
        pylab.imshow(data.reshape(28, 28), cmap=pylab.cm.gray_r, interpolation="nearest")
        pylab.title("%i" % label)
    pylab.show()

def ev_mnist(y, t):
    index_y = np.argmax(y, axis=1)
    index_t = np.argmax(t, axis=1)

    print(index_y[:20])
    print(index_t[:20])

    acc = index_y == index_t
    print(sum(acc), len(acc))

def foo(filename, dim=35):
    ret = []
    for line in open(filename):
        ret.append(float(line.rstrip()))
    
    ret = np.array(ret)
    ret = ret.reshape((-1, dim))

    return ret

def to_txt(file_recon="gen/recon.txt"):
    import subprocess

    parent_path = os.path.abspath(os.path.dirname(__file__))
    parent_path = parent_path[:-6]
    
    cmd = "x2x +fa gen/phrase.mgc > {0}".format(parent_path+file_recon)
    proc = subprocess.call(cmd, shell=True)

def plot_mcep(index=5, start=0, end=-1, dim=35, loc=3, file_recon="gen/recon.txt", t="a01", label="DRM"):
    import matplotlib.pyplot as plt
    import subprocess

    parent_path = os.path.abspath(os.path.dirname(__file__))
    parent_path = parent_path[:-6]
    
    file_mcep = parent_path + "gen/target/{0}.txt".format(t)

    #cmd = "x2x +fa gen/phrase.mgc > {0}".format(parent_path+file_recon)
    #proc = subprocess.call(cmd, shell=True)

    to_txt(file_recon)
    
    mcep = foo(file_mcep)
    recon = foo(file_recon)
    x = range(len(mcep))[start:end]
    
    mcep = split(mcep, index=index, start=start, end=end)
    recon = split(recon, index=index, start=start, end=end)
    
    plt.plot(x, mcep, "k", lw=1, label="Natural speech")
    plt.plot(x, recon, "r", lw=2, label=label)
    
    plt.xlim(start, start+len(x))
    plt.ylabel("{0}-th Mel-cepstrum".format(index))
    plt.xlabel("frame")
    plt.legend(loc=loc)
    plt.show()

def ev_ling(y, t, n=42):
    b1, b2, b3 = np.hsplit(y, [n, 2*n])
    y = np.c_[softmax(b1), softmax(b2), softmax(b3)]
    
    y = np.hsplit(y, [n, 2*n])
    t = np.hsplit(t, [n, 2*n])

    for i in range(len(y)):
        index_y = np.argmax(y[i], axis=1)
        index_t = np.argmax(t[i], axis=1)

        acc = index_y == index_t
        print(sum(acc), len(acc))

def std(x, mu=None, s=None, axis=0):
    #import scipy as sp
    #from sklearn.preprocessing import StandardScaler

    x = np.copy(x)

    if mu is None and s is None:
        mu = np.mean(x, axis=axis)
        s = np.std(x, axis=axis)
        
        zeros = np.where(s==0)
        for i in zeros:
            s[i] = 1.

        return (x - mu)/s, mu, s
    else:
        return (x - mu)/s

def std_inv(x, mu, s, axis=0):
    return x * s + mu

def normalize(y, vmin=None, vmax=None, begin=0.01, end=0.99, axis=0):
    y = np.copy(y, np.float16)

    if vmin is None:
        vmin = y.min(axis=axis)
        vmax = y.max(axis=axis)

        flag = 1

    if axis == None:
        if vmin != vmax:
            y = (y - vmin) * (end - begin) / (vmax - vmin) + begin
        else:
            for i in range(y.shape[0]):
                for j in range(y.shape[1]):
                     y[i][j] = (end + begin) / 2.
    elif axis == 0:
        y = (y - vmin) * (end - begin) / (vmax - vmin) + begin
    elif axis == 1:
        y = ((y.T - vmin) * (end - begin) / (vmax - vmin) + begin).T

    if flag == 0:
        return y
    else:
        return y, vmin, vmax

def normalize_inv(y, vmin, vmax, begin=0.01, end=0.99, axis=0):
    y = np.copy(y)
    
    if axis == None:
        if vmin != vmax:
            y = (y - begin) * (vmax - vmin) / (end - begin) + vmin
    elif axis == 0:
        y = (y - begin) * (vmax - vmin) / (end - begin) + vmin
    elif axis == 1:
        y = ((y.T - begin) * (vmax - vmin) / (end - begin) + vmin).T

    return y

def MCD(vt, vr):
    import math

    vt = np.array(vt)
    vr = np.array(vr)
    a = math.sqrt(2.) * 10. / math.log(10.)
    #a = 10. / math.log(10.)

    return a*np.sum(np.sqrt(np.sum(np.square(vt - vr), axis=1)))/vt.shape[0]

if __name__ == "__main__":
    
    y = np.array([[1, 2, 3], [3, 4, 5]])
    y = normalize(y)
    #print(y.shape)

    a = np.array([[1, 2, 3], [4, 5, 6]])
    print(a)

    a, vmin, vmax = normalize(a, axis=1)
    print(a)
    a = normalize_inv(a, vmin, vmax, axis=1)
    print(a)

    """
    a = np.array([[1, 2, 3], [4, 5, 6]])
    b = np.array([[2, 3, 4], [4, 5, 7]])

    print(MCD(a, b))
    """
