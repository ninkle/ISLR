# APPLIED EXERCISES FROM "INTRODUCTION TO STATISTICAL LEARNING WITH APPLICATIONS IN R

import numpy as np
import pandas as pd
from scipy import stats
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

def p_value(x, y):
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    return p_value

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

3. " Suppose we have a dataset with five predictors, X1 = GPA, X2 = IQ, X3 = Gender (1 for Female and 0 for Male),
     X4 = Interaction between GPA and IQ, X5 = Interaction between GPA and Gender. The response is starting salary
     after graduation (in thousands of dollars). Suppose we use least squares to fit the model and get β0 = 50,
     β1 = 20, β2 = 0.07, β3 = 35, β4 = 0.01, β5 = -10. "

     a. We see that with the given coefficients, our regression function takes the form:

        y = β0 + β1*X1 + β2*X2 + β3*X3 + β4*X4 + β5*X5

        If we hold IQ and GPA fixed, the the function reduces to

        y = β0 + β1 + β2 + β3*X3 + β4 + β5

        Substituting in the possible values for X3 (1 or 0),  we have

        y1 = β0 + β1 + β2 + β3 + β4 + β5
        i.e y = 50 + 20 + 0.07 + 35 + 0.01 - 10
        if the sample is female

        y2 = β0 + β1 + β2 + β4
        i.e y = 50 + 20 + 0.07 + 0.01
        if the sample is male

        Graphing the two functions, y1 and y2, where we allow for a variable to take the place of first IQ,
        there is no intersection of lines. When we allow a variable to take the place of GPA, we see that the two
        function lines intersect when GPA = 3.5. That is, when IQ and GPA are fixed, females typically earn more
        than males - except when GPA >= 3.5.


     b. For a female with IQ = 110, and GPA = 4.0,

        y = 50 + 20*(4.0) + 0.07*(110) + 35 + 0.01*(110*4.0) - 10(4.0*1)

        y = 137.1 thousand dollars.


     c. We notice that the coefficient term for GPA/IQ interaction is quite small. Rather than immediately discounting
        the GPA/IQ coefficient one which is not statistically significant, we recall that the value of IQ ranges from 0
        to upwards of 130, which is quite a bit larger than the scale of GPA (ranging from 0-4) and the female/male
        distinction which is a binary value of either 1 or 0. In order to accommodate for the discrepancy in ranges,
        the coefficient for both the singular IQ value and that for the GPA/IQ interaction value are quite a bit
        smaller in comparison to the remaining coefficients.

'''

'''# APPLIED EXERCISES
df = pd.read_csv("Auto.csv")
# print(df.head())

# drop rows with non-int placeholders
df = (df[df["horsepower"] != "?"])

Y = df["mpg"].as_matrix()  # a 1-dimensional numpy array of the response variable (mpg)
X = df["horsepower"].astype("int").as_matrix()  # a 1-dimensional numpy array of the predictor variable (horsepower)'''




'''X_lin = [i for i in range(0, 50)]
Y_lin = [(i*2) + 4 + np.random.uniform(0, 3) for i in X_lin]

print(Y_lin)'''






