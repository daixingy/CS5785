import os
import numpy as np
import csv
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import MultipleLocator
import random
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from pandas.core.frame import DataFrame

def read_mnist(path):
    '''
    read mnist dataset
    '''
    file = open(path, "r")
    reader = csv.reader(file)
    data = []
    for line in reader:
        if reader.line_num == 1:
            continue
        data.append(line)
    return data


def process(data, col):
    a = []
    data = np.array(data)
    for i in range(len(data)):
        a.append(data[i][col].tolist())
    attri_avg = []
    for j in range(len(a[0])):
        temp = 0
        count = 0
        for i in range(len(a)):
            if a[i][j] != 'NA':
                temp = temp + float(a[i][j])
                count += 1
            else:
                continue
        temp = temp / count
        attri_avg.append(temp)
    
    for i in range(len(a)):
        for j in range(len(a[i])):
            if (a[i][j] == 'NA'):
                a[i][j] = attri_avg[j]
            else:
                a[i][j] = float(a[i][j])
    return a

def f(X, theta):
    y_prime = X.dot(theta)
    return y_prime

def lsq_gradient(theta, X, y):
    a = 2 * (X.T.dot(X)).dot(theta) - 2 * X.T.dot(Y)
    return a

def least_square(theta, X, y):
    y_prime = f(X, theta)
    return np.linalg.norm(y_prime - y)

def train(X_train, y_train, threshold=1e-3, step_size=4e-1):
    threshold = 1e-3
    step_size = 4e-1
    theta, theta_prev = np.zeros(shape=(X_train.shape[1],1)), (np.zeros(shape=(X_train.shape[1],1)) + 0.01)
    opt_pts = [theta]
    opt_grads = []
    iter = 0
    while np.linalg.norm(theta - theta_prev) > threshold:
        if iter % 100 == 0:
            print('Iteration %d. LSQ: %.6f' % (iter, least_square(theta, X_train, y_train)))
        theta_prev = theta
        gradient = lsq_gradient(theta, X_train, y_train)
        theta = theta_prev - step_size * gradient
        opt_pts += [theta]
        opt_grads += [gradient]
        iter += 1

if __name__ == "__main__":
    path = "./dataset/house-prices-advanced-regression-techniques/train.csv"
    data = read_mnist(path)
    label = []
    for i in range(len(data)):
        label.append([])
        label[i].append(float(data[i][-1]))
    num_col = [2, 4, 5, 18, 19, 20, 21, 27, 35, 37, 38, 39, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 55, 57, 60, 62, 63, 68, 69, 70, 71, 72,76, 77, 78]
    # num_col = [2, 4, 5, 18, 19, 35, 37, 38, 39, 44]
    for i in range(len(num_col)):
        num_col[i] = num_col[i] - 1
    data = process(data, num_col)
    X = np.array(data)
    Y = np.array(label)
    reg = LinearRegression().fit(X, Y)
    print(reg.score(X, Y))
    clf = Ridge(alpha=1.0)
    clf.fit(X, Y)
    print(clf.score(X, Y))
    lasso = Lasso(alpha=1.0)
    lasso.fit(X, Y)
    print(lasso.score(X, Y))
    test_path = "./dataset/house-prices-advanced-regression-techniques/test.csv"
    data = read_mnist(test_path)
    data = process(data, num_col)
    X = np.array(data)
    Y = reg.predict(X)
    id = []
    answer = []
    for i in range(len(X)):
        id.append(i+1461)
        answer.append(Y[i][0])
    ex = {"Id":id,
            "SalePrice":answer}
    d=DataFrame(ex)
    d.to_csv("./house.csv", index=False)

    