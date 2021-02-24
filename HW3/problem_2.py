import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('TkAgg')
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from matplotlib.colors import ListedColormap
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
from sklearn.neighbors import KNeighborsRegressor

def preprocess(data):
    i = 0
    chosen = set()
    total = data.shape[0] * data.shape[1]
    sample = []
    while(i < 5000):
        index = int(random.random() * total)
        while (index in chosen):
            index = int(random.random() * total)
        chosen.add(index)
        i += 1
    sample_y = []
    for i in chosen:
        w = i//data.shape[1]
        h = i%data.shape[1]
        sample.append([w, h])
        sample_y.append(data[w][h]/255.0)
    sample_data = np.array(sample)
    sample_y = np.array(sample_y)
    return sample_data, sample_y

def preprocess_label(data):
    r = []
    g = []
    b = []
    for i in range(len(data)):
        r.append(data[i][0])
        g.append(data[i][1])
        b.append(data[i][2])
    return np.array(r), np.array(g), np.array(b)

def create_RF(n = 10):
    clf = RandomForestRegressor(n_estimators = n)
    return clf

def build_image(models, data, train_X, train_Y):
    p_data = []
    t_tain = train_X.tolist()
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            temp = [i,j]
            if temp in t_tain:
                continue
            p_data.append([i, j])
    p_data = np.array(p_data)
    print(p_data.shape)
    r = models[0].predict(p_data)
    g = models[1].predict(p_data)
    b = models[2].predict(p_data)
    pic = []
    t_p_data = p_data.tolist()
    print(type(train_Y))
    train_Y = train_Y.tolist()
    for i in range(data.shape[0]):
        # print(i)
        pic.append([0] * data.shape[1])
    for i in range(len(t_p_data)):
        # print(i)
        w = t_p_data[i][0]
        h = t_p_data[i][1]
        pic[w][h]=[int(r[i]*255), int(g[i]*255), int(b[i]*255)]
    for i in range(len(train_X)):
        # print(i)
        w = train_X[i][0]
        h = train_X[i][1]
        pic[w][h] = [int(train_Y[i][0]*255), int(train_Y[i][1]*255), int(train_Y[i][2]*255)]
    pic = np.array(pic)
    print(pic.shape)
    pic = np.array(pic)
    fig = plt.figure()
    plt.imshow(pic)
    plt.show()
    fig.savefig("./mo2.jpg")


def exp_image(models, data, train_X, train_Y, pic_name):
    p_data = []
    t_tain = train_X.tolist()
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            temp = [i,j]
            if temp in t_tain:
                continue
            p_data.append([i, j])
    p_data = np.array(p_data)
    print(p_data.shape)
    r = models[0].predict(p_data)
    g = models[1].predict(p_data)
    b = models[2].predict(p_data)
    pic = []
    t_p_data = p_data.tolist()
    print(type(train_Y))
    train_Y = train_Y.tolist()
    for i in range(data.shape[0]):
        # print(i)
        pic.append([0] * data.shape[1])
    for i in range(len(t_p_data)):
        # print(i)
        w = t_p_data[i][0]
        h = t_p_data[i][1]
        pic[w][h]=[int(r[i]*255), int(g[i]*255), int(b[i]*255)]
    for i in range(len(train_X)):
        # print(i)
        w = train_X[i][0]
        h = train_X[i][1]
        pic[w][h] = [int(train_Y[i][0]*255), int(train_Y[i][1]*255), int(train_Y[i][2]*255)]
    pic = np.array(pic)
    print(pic.shape)
    pic = np.array(pic)
    fig = plt.figure()
    plt.imshow(pic)
    # plt.show()
    fig.savefig(pic_name)


def set_RF(n_estimators, depth):
    clf = RandomForestRegressor(n_estimators = n_estimators, max_depth=depth)
    return clf

def experimentation_i(data, train_X, train_Y, r, g, b):
    depth = [1,2,3,5,10,15]
    for i in depth:
        f_red = set_RF(1,i)
        f_green = set_RF(1,i)
        f_blue = set_RF(1,i)
        f_red.fit(train_X, r)
        f_green.fit(train_X, g)
        f_blue.fit(train_X, b)
        models = [f_red, f_green, f_blue]
        pic_name = "./exp_mo_" + str(i) + ".png"
        exp_image(models, data, train_X, train_Y, pic_name)

def experimentation_ii(data, train_X, train_Y, r, g, b):
    est = [1,3,5,10,100]
    for i in est:
        f_red = set_RF(i,7)
        f_green = set_RF(i,7)
        f_blue = set_RF(i,7)
        f_red.fit(train_X, r)
        f_green.fit(train_X, g)
        f_blue.fit(train_X, b)
        models = [f_red, f_green, f_blue]
        pic_name = "./exp_mo_est_" + str(i) + ".png"
        exp_image(models, data, train_X, train_Y, pic_name)

def KNN_model():
    clf = KNeighborsRegressor(n_neighbors=1)
    return clf

def experimentation_iii(data, train_X, train_Y, r, g, b):
    f_red = KNN_model()
    f_green = KNN_model()
    f_blue = KNN_model()
    f_red.fit(train_X, r)
    f_green.fit(train_X, g)
    f_blue.fit(train_X, b)
    models = [f_red, f_green, f_blue]
    pic_name = "./KNN.png"
    exp_image(models, data, train_X, train_Y, pic_name)


if __name__ == "__main__":
    path = "./data_set/Mona.jpg"
    img = mpimg.imread(path)
    # plt.figure()
    # plt.imshow(img)
    # plt.show()
    train_X, train_Y = preprocess(img)
    # print(train_X.shape)
    r, g, b = preprocess_label(train_Y)
    f_red = create_RF()
    f_green = create_RF()
    f_blue = create_RF()
    f_red.fit(train_X, r)
    f_green.fit(train_X, g)
    f_blue.fit(train_X, b)
    models = [f_red, f_green, f_blue]
    print(img.shape)
    # build_image(models, img, train_X, train_Y)
    experimentation_iii(img, train_X, train_Y, r, g, b)

    
    