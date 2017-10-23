# APPLIED EXERCISES FROM "INTRODUCTION TO STATISTICAL LEARNING WITH APPLICATIONS IN R

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Some functions written in order to complete the applied questions section

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


'''
CONCEPTUAL EXERCISES

1. " Describe the null hypothesis to which the p-values in Table 3.4 correspond. Explain what conclusions
     you can draw based on those p-values. "

     TABLE 3.4
                Coefficient  Std. Error  T-Statistic  P-Values
     Intercept     2.939        0.3119       9.42      < 0.0001

     TV            0.046        0.0014       32.81     < 0.0001

     Radio         0.189        0.0086       21.89     < 0.0001

     Newspaper     -0.001       0.0059       -0.18     0.8599


     We see from the p-values in Table 3.4 that we can reject the null hypothesis (i.e. that β(i) = 0)
     for the Intercept, TV, and Radio, as their respective p-values are sufficiently small. In the case
     of Newspaper, however, we can assume based off of the largeness of the p-value that the data has
     no statistical significance.

2. " Carefully explain the difference between the KNN classifier and KNN regression methods. "


     We recall from Chapter 2 that the KNN classifier identifies, for some arbitrarily selected integer K and
     some test observation x0, the K points in the training data that are closest to x0, which are represented
     by N0. For each class j corresponding to some subset of N0, we calculate the conditional probability of j
     as the fraction of the points in N0 whose response value is equal to j:

            i.e. Pr(Y = j|X = x0) = 1/K * np.sum(I(y = j))

     Invoking Bayes rule, we then assign to the test observation x0 whichever class j yields the highest probability.

     In contrast, the KNN regression method, takes the average of each the responses in N0:

            i.e. f(x0) = 1/K * np.sum(y)

     The predicted response for the test observation x0 is this average.

'''


# APPLIED EXERCISES
df = pd.read_csv("Auto.csv")
# print(df.head())

# drop rows with non-int placeholders
df = (df[df["horsepower"] != "?"])

Y = df["mpg"].as_matrix()  # a 1-dimensional numpy array of the response variable (mpg)
X = df["horsepower"].astype("int").as_matrix()  # a 1-dimensional numpy array of the predictor variable (horsepower)







