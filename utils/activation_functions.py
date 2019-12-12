#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@file: activation_functions.py
@time: 2019-12-10 16:19
"""
import numpy as np


def relu(X):
    return np.maximum(X, 0)


def softmax(X):
    X_exp = np.exp(X)
    partition = np.sum(X_exp)
    return X_exp / partition  # 这里应用了广播机制


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanh(x):
    return (1 - np.exp(-x)) / (1 + np.exp(-x))
