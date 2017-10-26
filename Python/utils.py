# APPLIED EXERCISES FROM "INTRODUCTION TO STATISTICAL LEARNING WITH APPLICATIONS IN R

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Some functions written in order to complete the applied questions section

def get_residuals(y_true, y_pred):
    return (y_true-y_pred)

def rss(y_true, y_pred):
    return np.sum((y_true-y_pred)**2)


def rse(y_true, y_pred, p=1):  # designed for the simple lin reg setting, where p = 1
    return np.sqrt((1/(len(y_true)-p-1))*rss(y_true, y_pred))


def tss(y):
    return np.sum((y-np.mean(y))**2)


def var(x):  # takes a numpy array as input, returns a scalar number
    return np.mean((x-np.mean(x))**2)


def standard_error(x):  # takes a numpy array as input, returns a scalar number
    return var(x)/2  # tells us how accurate the sample mean is as an estimate of the population mean


def least_squares(X, Y):  # calculate coefficients in a simple lin reg setting
    b1 = np.sum((X-np.mean(X)) * (Y-np.mean(Y))) / np.sum(((X-np.mean(X))**2))
    b0 = np.mean(Y) - b1*np.mean(X)
    return b1, b0


def simple_lin_reg(x, y):
    b1, b0, = least_squares(x, y)
    y_pred = x*b1 + b0
    return y_pred


def standard_error_b0(x, y_true, y_pred):
    variance = rse(y_true, y_pred) ** 2
    n = len(x)
    return variance * ( (1/n) + ( (np.mean(x) ** 2 ) / np.sum( (x-np.mean(x)) ** 2 )))


def standard_error_b1(x, y_true, y_pred):
    variance = rse(y_true, y_pred) ** 2
    return variance / (np.sum( ((x - np.mean(x)) ** 2 )))


def r2(x, y):
    y_pred = simple_lin_reg(x, y)
    return 1 - (rss(y, y_pred) / tss(y))


def residuals(x, y):  # to analyze residuals for linearity/non-lin, we plot the residuals against the fitted values
    y_pred = simple_lin_reg(x, y)
    return (y - y_pred)


def f_statistic(y_true, y_pred, p=1):
    return ( (tss(y_true) - rss(y_true, y_pred)) / p) / (rss(y_true, y_pred) / (len(y_true)-p-1))


def correlation(x, y):
    num = np.sum( (x - np.mean(x)) * (y - np.mean(y)) )
    denom =  np.sqrt(np.sum ((x - np.mean(x))**2)) * (np.sqrt((np.sum((y-np.mean(y))**2))))
    return num/denom


def f_statistic_mul(x, y, coefficients, intercept, p):
    y_pred = np.dot(x,coefficients) + intercept
    return f_statistic(y, y_pred, p)

def t_statistic(beta, se_beta):
    return (beta - 0) / se_beta



