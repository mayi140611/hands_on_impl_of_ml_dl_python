from __future__ import division
import numpy as np
from sklearn.metrics import accuracy_score


def square_loss(y_true, y_hat):
    """
    回归问题的loss function
    每一个样本的预测值用一个标量表示
    :param y_true: np.array([0.2,0.6,0.8])
    :param y_hat: np.array([0.18, 0.5, 0.7])
    :return:
    """
    return 0.5 * np.mean(np.power((y_true - y_hat), 2))


def log_loss_binary(y_true, y_hat):
    """
    二分类loss function. 每一个样本的预测值用一个标量表示
    :param y_true: np.array([1, 0, 1])
    :param y_hat: np.array([0.18, 0.5, 0.7])
    :return:
    """
    # Avoid division by zero
    y_hat = np.clip(y_hat, 1e-15, 1 - 1e-15)
    return - y_true * np.log(y_hat) - (1 - y_true) * np.log(1 - y_hat)


def cross_entropy_loss(y_true, y_hat):
    """
    multi分类loss function. 每一个样本的预测值用一个one-hot vector表示
    :param y_true: np.array([(1, 0, 0), (1, 0, 0), (1, 0, 0)])
    :param y_hat: np.array([(0.18, 0.5, 0.7), (0.18, 0.5, 0.7),(0.18, 0.5, 0.7),])
    :return:
    """
    # Avoid division by zero
    y_hat = np.clip(y_hat, 1e-15, 1 - 1e-15)
    # 注意计算是由于-y_true * np.log(y_hat) element-wise，
    # 每一行只有一个值非0，所以按照cross_entropy_loss有两个\sum，可以合并成1个
    return np.sum(-y_true * np.log(y_hat))/len(y_true)
