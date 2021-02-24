from pylab import *
import matplotlib as mpl
mpl.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
from scipy.ndimage import filters
import urllib
from numpy import random
from keras.utils import to_categorical
import pickle as pk
import os
import scipy.io as scio
import cv2
import scipy.optimize as opt

def read_data(path):
    '''
    read mnist data
    '''
    data = scio.loadmat(path)
    return data

def one_hot(y_):
    return to_categorical(y_)

def get_pictures(filepath):
    path = os.listdir(filepath)
    path.remove('.DS_Store')
    pictures = []
    actor_name = []
    # print(path)
    count = 0
    for actor in path:
        actor_path = filepath + actor + "/"
        actor_path = os.listdir(actor_path)
        # print(actor_path)
        for i in actor_path:
            pic = filepath + actor + "/" + i
            p = mpimg.imread(pic)
            pictures.append(p)
            actor_name.append(count)
        count += 1
    # return np.array(pictures), one_hot(np.array(actor_name))
    return np.array(pictures), np.array(actor_name)

def rgb2gray(rgb):
    '''Return the grayscale version of the RGB image rgb as a 2D numpy array
    whose range is 0..1
    Arguments:
    rgb -- an RGB image, represented as a numpy array of size n x m x 3. The
    range of the values is 0..255
    '''
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    # r, g, b = rgb[0], rgb[1], rgb[2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray/255.

def resize_pictures(train_X):
    new_train_X = []
    for i in range(len(train_X)):
        '''
        resize all pictures
        '''
        train_X[i] = cv2.resize(train_X[i], dsize=(32, 32), interpolation=cv2.INTER_CUBIC)
        # train_X[i] = imresize(train_X[i], (200, 200))
        if len(train_X[i].shape) == 2:
            train_X[i] = np.reshape(train_X[i],(1024))
            t = train_X[i].tolist()
            new_train_X.append(t)
            continue
        '''
        transfer all pictures into gray
        '''
        train_X[i] = rgb2gray(train_X[i])
        train_X[i] = np.reshape(train_X[i],(1024))
        t = train_X[i].tolist()
        new_train_X.append(t)
    new_train_X = np.array(new_train_X)
    # print(new_train_X.shape)
    return new_train_X

def split_and_preprocess(train_X, train_Y):
    total_num = len(train_X)
    validation_set = set()
    test_set = set()
    print(total_num)
    while(len(validation_set) < int(0.1*total_num)):
        validation_index = int(total_num*(random.random()))
        test_index = int(total_num*(random.random()))
        while(validation_index in validation_set or validation_index in test_set):
            validation_index = int(total_num*(random.random()))
        validation_set.add(validation_index)
        while(test_index in validation_set or test_index in test_set):
            test_index = int(total_num*(random.random()))
        test_set.add(test_index)
    train_set = set()
    for i in range(total_num):
        if (i in validation_set or i in test_set):
            continue
        train_set.add(i)
    train = []
    train_label = []
    validation = []
    validation_label = []
    test = []
    test_label = []
    for i in range(total_num):
        if i in train_set:
            train.append(train_X[i])
            train_label.append(train_Y[i])
        elif i in validation_set:
            validation.append(train_X[i])
            validation_label.append(train_Y[i])
        else:
            test.append(train_X[i])
            test_label.append(train_Y[i])
    return np.array(train), np.array(train_label), np.array(validation), np.array(validation_label), np.array(test), np.array(test_label)

def sigmoid(z):
    # print(1  / (1 + np.exp(-z)))
    return 1  / (1 + np.exp(-z))

def logistic_regression_cost_function(theta, X, y, lamda):
    m = X.shape[0]
    cost = 0
    grad = np.zeros(theta.shape)
    hypothesis = sigmoid(np.dot(X, theta))
    reg_theta = theta[1:]
    # cost = np.sum(-y * np.log(hypothesis) - (1 - y) * np.log(1 - hypothesis)) / m + (lamda / (2 * m)) * np.sum(reg_theta * reg_theta)
    cost = np.sum(-y * np.log(hypothesis) - (1 - y) * np.log(1 - hypothesis)) / m
    normal_grad = (np.dot(X.T, hypothesis - y) / m).flatten()
    grad[0] = normal_grad[0]
    # grad[1:] = normal_grad[1:] + reg_theta * (lamda / m)
    grad[1:] = normal_grad[1:]
    return cost, grad

# def bi_classifier_test():
#     theta = np.array([-2, -1, 1, 2])
#     X = np.c_[np.ones(5), np.arange(1, 16).reshape((3, 5)).T/10]
#     y = np.array([1, 0, 1, 0, 1])
#     lamda = 3
#     cost, grad = logistic_regression_cost_function(theta, X, y, lamda)
#     np.set_printoptions(formatter={'float': '{: 0.6f}'.format})
#     print('Cost: {:0.7f}'.format(cost))
#     print('Expected cost: 2.534819')
#     print('Gradients:\n{}'.format(grad))
#     print('Expected gradients:\n[ 0.146561 -0.548558 0.724722 1.398003]')

def multi_class_cost_function(theta, X, y):
    m = y.shape[0]
    cost = 0
    grad = np.zeros(theta.shape)
    loss = []
    for i in range(m):
        s = np.power(np.array(np.theta.T.dot(X[i]) - y[i]), 2).sum()
        loss.append(s)
    loss = np.array(loss).sum()

def One_Vs_All(X, y, num_labels, lamda):
    print(X.shape)
    (m, n) = X.shape
    # all_theta = np.zeros((num_labels, n+1))
    all_theta = np.zeros((num_labels, n))
    # X = np.c_[np.ones(m), X]
    print(X.shape)
    print(X)
    for i in range(0, num_labels):
        print('Optimizing for actors\' faces {}...'.format(i))
        initial_theta = np.zeros((n, 1))
        y_i = np.array([1 if j == i else 0 for j in y])
        y_i = y
        def cost_func(t):
            return logistic_regression_cost_function(t, X, y_i, lamda)[0]
        def grad_func(t):
            return logistic_regression_cost_function(t, X, y_i, lamda)[1]
        theta, *unused = opt.fmin_cg(cost_func, fprime=grad_func, x0=initial_theta, maxiter=100, disp=False, full_output=True)
        print('Done')
        all_theta[i] = theta
    return all_theta

def num_labs(x):
    s = set()
    for i in x:
        s.add(i)
    return len(s)


from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

if __name__ == "__main__":
    # bi_classifier_test()
    filePath = 'actors/faces/'
    train_X, train_Y = get_pictures(filePath)
    # # print(train_X[200].shape)
    train_X = resize_pictures(train_X)
    train, train_label, validation, validation_label, test, test_label = split_and_preprocess(train_X, train_Y)
    num = num_labs(train_label)
    model = LogisticRegression()
    ovr = OneVsRestClassifier(model)
    ovr.fit(train, train_label)
    # make predictions
    yhat = ovr.predict(train)
    # theta = One_Vs_All(train, train_label, num, 0.1)
    print(yhat)
    print(train_label)
    print(yhat - train_label)