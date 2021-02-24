import os
import numpy as np
import csv
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import MultipleLocator
import random
from sklearn.metrics import confusion_matrix
from scipy.spatial import distance as matrix_distance

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


def process_data(data):
    for i in range(len(data)):
        for j in range(len(data[i])):
            data[i][j] = float(data[i][j])
    return data

def data_dic(data):
    data = process_data(data)
    dic = {}
    dat = []
    for i in range(len(data)):
        dic[tuple(data[i][1:])] = data[i][0]
        dat.append(data[i][1:])
    return dic, dat


class Node:
    def __init__(self, data, split=0, left=None, right=None):
        self.data = data
        self.split = split
        self.left = left
        self.right = right

class KDTree:
    def create(self, data, split):
        if (len(data) == 0):
            return None
        dim = len(data[0])
        data = sorted(data, key=lambda x: x[split])
        pivotal = int(len(data) / 2)
        mid_point = data[pivotal]
        root = Node(mid_point, split)
        root.left = self.create(data[: pivotal], (split+1)%dim)
        root.right = self.create(data[pivotal+1:], (split+1)%dim)
        return root
    
    def __init__(self, data):
        self.root = self.create(data=data, split=0)
    
    def KNN(self, data_point, dic, k=1):
        k_nearest_data = []
        for i in range(k):
            k_nearest_data.append([-1, None])
        self.k_nearest_data = np.array(k_nearest_data)

        def search(point, node, split, dic):
            if (node != None):
                d = point[split] - node.data[split]
                dim = len(point)
                if d < 0:
                    search(point, node.left, (split+1)%dim, dic)
                else:
                    search(point, node.right, (split+1)%dim, dic)
                distance = np.linalg.norm(point - node.data)
                for i, dd in enumerate(self.k_nearest_data):
                    if (dd[0] < 0 or dd[0] > distance):
                        self.k_nearest_data = np.insert(self.k_nearest_data, i, [distance, dic[tuple(node.data)]], axis=0)
                        self.k_nearest_data = self.k_nearest_data[:-1]
                        break
                # compare the other side of KDTree
                n = list(self.k_nearest_data[:, 0]).count(-1)
                # there may exist data points in other side which is closer to the target
                if self.k_nearest_data[-n-1, 0] > abs(d):
                    if d >= 0:
                        search(point, node.left, (split+1)%dim, dic)
                    else:
                        search(point, node.right, (split+1)%dim, dic)
        split=0
        search(data_point, self.root, split, dic)
        label_stat = [0,0,0,0,0,0,0,0,0,0]
        for i in range(len(self.k_nearest_data)):
            if self.k_nearest_data[i][1] == None:
                continue
            # print(self.k_nearest_data[i][1])
            label_stat[int(self.k_nearest_data[i][1])] += 1
        result = -1
        answer = -1
        for i in range(len(label_stat)):
            if label_stat[i] > result:
                result = label_stat[i]
                answer = i
        return answer

def split_data(data, p=0.15):
    split = []
    max_len = int(p * len(data))
    while(len(split) < max_len):
        index = int(random.random() * len(data))
        split.append(data[index])
        temp = data[index]
        data.remove(temp)
    return split, data
        
def simple_split(data, p=0.5):
    split = []
    max_len = int(p * len(data))
    split = data[0:max_len+1]
    data = data[max_len+1:]
    return split, data

def get_label(data):
    label = []
    for i in range(len(data)):
        label.append(int(data[i][0]))
    return label


def problem_1_h(data, KK):
    split, data = split_data(data, 0.5)
    data = process_data(data)
    dic, dat = data_dic(data)
    # build KDTree
    K = KDTree(np.array(dat))
    split_dic, split_dat = data_dic(split)
    # split data
    K_holdout_list = []
    K_train_list = []
    count = 0
    for i in range(len(KK)):
        accuracy = 0
        train_acc = 0
        # training accuracy
        # for t in range(len(data)):
        #     print("process:{0}%".format(round((count) * 100 / (len(KK)*42000))), end="\r")
        #     count += 1
        #     predict_label = K.KNN(np.array(dat[t]), dic, KK[i])
        #     # print(predict_label == int(data[t][0]))
        #     if (predict_label == int(data[t][0])):
        #         train_acc += 1
        # train_acc = float(train_acc) / float(len(data))
        # K_train_list.append(train_acc)

        # holdout accuracy
        for j in range(len(split)):
            print("process:{0}%".format(round((count) * 100 / (len(KK)*50))), end="\r")
            count += 1
            # predict_label = simple_KNN(split[j], data, K[i])
            predict_label = K.KNN(np.array(split_dat[j]), dic, KK[i])
            # print(predict_label == int(split[j][0]))
            if (predict_label == int(split[j][0])):
                accuracy += 1
        accuracy = float(accuracy) / float(len(split))
        K_holdout_list.append(accuracy)
    # print(K_train_list)
    print(K_holdout_list)
    plt.figure()
    # plt.plot(range(len(K_train_list)), K_train_list)
    # plt.savefig("./KNN_train.png")

    plt.figure()
    plt.plot(range(len(K_holdout_list)), K_holdout_list)
    plt.savefig("./hw01_written3.png")

if __name__ == "__main__":
    path = "./dataset/digit-recognizer/part_train.csv"
    data = read_mnist(path)
    Bin_data = []
    for i in range(len(data)):
        if data[i][0] == '0' or data[i][0] == '1':
            Bin_data.append(data[i])
    data = Bin_data
    K = []
    for i in range(101):
        K.append(i+1)
    problem_1_h(data, K)