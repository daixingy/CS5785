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

def show_each_digit(data):
    script = []
    digit_set = set()
    matplotlib.use('TkAgg')
    figure, axis = plt.subplots(nrows=1, ncols=10, sharex='all', sharey='all')
    for i in range(0, len(data)):
        script.append([])
        for j in range(1, len(data[i]), 1):
            script[i].append(float(data[i][j]) / 255.0)
            # reshape data
        script[i] = np.array(script[i]).reshape((28, 28))
        if data[i][0] not in digit_set:
            axis[int(data[i][0])].imshow(script[i], cmap='Greys', interpolation='nearest')
            digit_set.add(data[i][0])
        else:
            continue
        if (len(digit_set) == 10):
            break
    plt.tight_layout()
    plt.savefig("./digits.png")
    plt.show()



def problem_1_c(data):
    label = []
    labels_stat = [0,0,0,0,0,0,0,0,0,0]
    for i in range(len(data)):
        label.append(data[i][0])
        labels_stat[int(data[i][0])] += 1
    label = np.sort(np.array(label))
    # print(label)
    plt.hist(label, weights=np.ones(len(label)) / len(label), edgecolor='k', alpha=0.5)
    plt.savefig("./histogram.png")


def problem_1_d(data):
    sample = []
    sample_sign = []
    sample_rank = []
    digit_set = set()
    # current = []
    # pick sample from data set
    for i in range(0, len(data)):
        current = []
        for j in range(1, len(data[i]), 1):
            current.append(float(data[i][j]))
            # reshape data
        label = int(data[i][0])
        if label not in digit_set:
            sample.append(current)
            sample_sign.append(label)
            sample_rank.append(i)
            digit_set.add(label)
        else:
            continue
        if (len(digit_set) == 10):
            break
    nearest_data = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
    nearest_label = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
    nearest_attribute = [[],[],[],[],[],[],[],[],[],[]]
    for j in range(len(sample)):
        nearest_distance = float('inf')
        for i in range(len(data)):
            data_attribute = []
            if (i == sample_rank[j]):
                continue
            for k in range(1, len(data[i]), 1):
                data_attribute.append(float(data[i][k]))
            data_label = int(data[i][0])
            sample_target = np.array(sample[j])
            data_attribute = np.array(data_attribute)
            # L2 distance
            distance = np.linalg.norm(sample_target-data_attribute)
            if distance<=nearest_distance:
                nearest_data[j] = i
                nearest_label[j] = data_label
                nearest_attribute[j] = data_attribute
                nearest_distance = distance
    print("Sample label:")
    print(sample_sign)
    print("Nearest Neighbor label:")
    print(nearest_label)
    print("Sample number in data set:")
    print(sample_rank)
    print("Nearest number in data set:")
    print(nearest_data)


def problem_1_e(data):
    digits_1 = []
    digits_0 = []
    for i in range(len(data)):
        if data[i][0] == '0':
            digits_0.append(data[i][1:])
        if data[i][0] == '1':
            digits_1.append(data[i][1:])
        else:
            continue
    digits_0 = process_data(digits_0)
    digits_1 = process_data(digits_1)
    digits_0 = np.array(digits_0)
    digits_1 = np.array(digits_1)

    impostor_distance = []
    genuine_distance = []
    distance_0 = distance.cdist(digits_0, digits_0, 'euclidean')
    distance_1 = distance.cdist(digits_1, digits_1, 'euclidean')
    distance_01 = distance.cdist(digits_0, digits_1, 'euclidean')
    distance_0 = np.reshape(distance_0, (1,-1)).tolist()[0]
    distance_1 = np.reshape(distance_1, (1,-1)).tolist()[0]
    distance_01 = np.reshape(distance_01, (1,-1)).tolist()[0]
    genuine_distance = distance_0 + distance_1
    impostor_distance = distance_01 + distance_01
    bins = range(0, 5000, 10)
    plt.hist(impostor_distance, bins=bins, color = 'skyblue', edgecolor='skyblue', alpha=0.05)
    plt.hist(genuine_distance, bins=bins, color = 'coral', edgecolor='coral', alpha=0.05)
    plt.savefig("./one_zero.png")

def problem_1_f(data):
    digits_1 = []
    digits_0 = []
    for i in range(len(data)):
        if data[i][0] == '0':
            digits_0.append(data[i][1:])
        if data[i][0] == '1':
            digits_1.append(data[i][1:])
        else:
            continue
    digits_0 = process_data(digits_0)
    digits_1 = process_data(digits_1)
    digits_0 = np.array(digits_0)
    digits_1 = np.array(digits_1)

    impostor_distance = []
    genuine_distance = []
    distance_0 = distance.cdist(digits_0, digits_0, 'euclidean')
    distance_1 = distance.cdist(digits_1, digits_1, 'euclidean')
    distance_01 = distance.cdist(digits_0, digits_1, 'euclidean')
    distance_0 = np.reshape(distance_0, (1,-1)).tolist()[0]
    distance_1 = np.reshape(distance_1, (1,-1)).tolist()[0]
    distance_01 = np.reshape(distance_01, (1,-1)).tolist()[0]
    genuine_distance = distance_0 + distance_1
    impostor_distance = distance_01 + distance_01
    roc_distance = 0
    max_distance = 0
    for i in range(len(genuine_distance)):
        if genuine_distance[i] > max_distance:
            max_distance = genuine_distance[i]
    TPR = []
    FPR = []
    for roc_distance in range(0, round(max_distance), 150):
        tpr = 0
        fpr = 0
        print("process:{0}%".format(round((roc_distance) * 100 / max_distance)), end="\r")
        for t in range(len(genuine_distance)):
            if genuine_distance[t] < roc_distance:
                tpr += 1
        for f in range(len(impostor_distance)):
            if impostor_distance[f] < roc_distance:
                fpr += 1
        tpr = tpr / len(genuine_distance)
        fpr = fpr / len(impostor_distance)
        TPR.append(tpr)
        FPR.append(fpr)
    plt.plot(FPR, TPR)
    plt.savefig("./ROC.png")



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

def simple_KNN(data_point, data, k):
    k_nearest_data = []
    predict_point = np.array(data_point[1:])
    # record k nearest data point
    for i in range(len(data)):
        compare_data = data[i][1:]
        distance = np.linalg.norm(np.array(compare_data) - predict_point)
        current = []
        current.append(distance)
        current.append(int(data[i][0]))
        k_nearest_data.append(current)
    k_nearest_data = np.array(k_nearest_data)[np.argsort(np.array(k_nearest_data)[:,0])]
    k_nearest_data = k_nearest_data.tolist()
    k_nearest_data = k_nearest_data[:k]
    label_stat = [0,0,0,0,0,0,0,0,0,0]
    for i in range(len(k_nearest_data)):
        label_stat[int(k_nearest_data[i][1])] += 1
    result = -1
    answer = -1
    print(label_stat)
    for i in range(len(label_stat)):
        if label_stat[i] > result:
            result = label_stat[i]
            answer = i
    return answer

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
            label_stat[int(self.k_nearest_data[i][1])] += 1
        result = -1
        answer = -1
        for i in range(len(label_stat)):
            if label_stat[i] > result:
                result = label_stat[i]
                answer = i
        return answer

def cdist_KNN(data, KK):
    split, data = simple_split(data, 0.15)
    data = process_data(data)
    split = process_data(split)
    data_lab = get_label(data)
    dic, dat = data_dic(data)
    split_dic, split_dat = data_dic(split)
    total_label = []
    train_label = []

    for i in range(len(split)):
        total_label.append(data_lab)
    for i in range(len(dat)):
        train_label.append(data_lab)

    K_h = []
    K_t = []
    count = 0
    y_true = []
    y_pred = []
    for i in range(len(KK)):
        # holdout data
        distance = matrix_distance.cdist(split_dat, dat, 'euclidean')
        index = np.argsort(np.array(distance)[0:,])
        index = np.split(index, (KK[i],), axis=1)[0]
        total_label = np.array(total_label)
        # label = []
        holdout_predict = []
        for j in range(len(total_label)):
            print("process:{0}%".format(round((count) * 100 / (6300 * len(KK)))), end="\r")
            count += 1
            temp_label = ((np.array(total_label)[j])[index[j]]).tolist()
            label_stat = [0,0,0,0,0,0,0,0,0,0]
            # temp_label = label[j]
            for t in range(len(temp_label)):
                current_t = int(temp_label[t])
                label_stat[current_t] += 1
            result = -1
            answer = -1
            # print(label_stat)
            for t in range(len(label_stat)):
                if label_stat[t] > result:
                    result = label_stat[t]
                    answer = t
            y_pred.append(answer)
            y_true.append(int(split[j][0]))
            if (answer == int(split[j][0])):
                # print(True)
                holdout_predict.append(1)
            else:
                # print(False)
                holdout_predict.append(0)
        hold_acc = np.array(holdout_predict).sum() / float(len(holdout_predict)) 
        K_h.append(hold_acc)
    print(confusion_matrix(y_true, y_pred))
        
        

def split_data(data, p=0.15):
    split = []
    max_len = int(p * len(data))
    while(len(split) < max_len):
        index = int(random.random() * len(data))
        split.append(data[index])
        temp = data[index]
        data.remove(temp)
    return split, data
        
def simple_split(data, p=0.15):
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
    split, data = split_data(data, 0.15)
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
            print("process:{0}%".format(round((count) * 100 / (len(KK)*6300))), end="\r")
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
    plt.savefig("./KNN_hold_out.png")

def problem_1_i(data, KK):
    split, data = simple_split(data, 0.15)
    data = process_data(data)
    split = process_data(split)
    data_lab = get_label(data)
    dic, dat = data_dic(data)
    split_dic, split_dat = data_dic(split)
    total_label = []
    train_label = []

    for i in range(len(split)):
        total_label.append(data_lab)
    for i in range(len(dat)):
        train_label.append(data_lab)

    K_h = []
    K_t = []
    count = 0
    y_true = []
    y_pred = []
    for i in range(len(KK)):
        # holdout data
        distance = matrix_distance.cdist(split_dat, dat, 'euclidean')
        index = np.argsort(np.array(distance)[0:,])
        index = np.split(index, (KK[i],), axis=1)[0]
        total_label = np.array(total_label)
        # label = []
        holdout_predict = []
        for j in range(len(total_label)):
            print("process:{0}%".format(round((count) * 100 / (6300 * len(KK)))), end="\r")
            count += 1
            temp_label = ((np.array(total_label)[j])[index[j]]).tolist()
            label_stat = [0,0,0,0,0,0,0,0,0,0]
            # temp_label = label[j]
            for t in range(len(temp_label)):
                current_t = int(temp_label[t])
                label_stat[current_t] += 1
            result = -1
            answer = -1
            # print(label_stat)
            for t in range(len(label_stat)):
                if label_stat[t] > result:
                    result = label_stat[t]
                    answer = t
            y_pred.append(answer)
            y_true.append(int(split[j][0]))
            if (answer == int(split[j][0])):
                # print(True)
                holdout_predict.append(1)
            else:
                # print(False)
                holdout_predict.append(0)
        hold_acc = np.array(holdout_predict).sum() / float(len(holdout_predict)) 
        K_h.append(hold_acc)
    print(confusion_matrix(y_true, y_pred))

def problem_1_j(data, test_data):
    data = process_data(data)
    dic, dat = data_dic(data)
    # build KDTree
    K = KDTree(np.array(dat))
    test_data = process_data(test_data)
    answer = []
    id = []
    for i in range(len(test_data)):
        print("process:{0}%".format(round((i) * 100 / len(test_data))), end="\r")
        id.append(i+1)
        answer.append(K.KNN(np.array(test_data[i]), dic, 6))
    ex = {"ImageId":id,
            "Label":answer}
    d=DataFrame(ex)
    d.to_csv("./result.csv", index=False)

if __name__ == "__main__":
    path = "./dataset/digit-recognizer/train.csv"
    data = read_mnist(path)
    # split, data = split_data(data)
    # print(len(data))
    # show_each_digit(data)
    # problem_1_c(data)
    # problem_1_d(data)
    # problem_1_e(data)
    # problem_1_f(data)
    # K = [2,3,4,5,6,7,8]
    # problem_1_h(data, K)
    # cdist_KNN(data, [2])
    # print(problem_1_g(data))