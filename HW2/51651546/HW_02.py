import os
import numpy as np
import math
import csv
import random
from pandas.core.frame import DataFrame
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.metrics import f1_score
import heapq
import string
import re
from pandas.core.frame import DataFrame
import matplotlib.pyplot as plt

def read_data(path):
    '''
    read nlp data
    '''
    file = open(path, "r")
    reader = csv.reader(file)
    data = []
    for line in reader:
        if reader.line_num == 1:
            continue
        data.append(line)
    return data

def stat_train_test_data_point(train, test):
    train_point = len(train)
    test_point = len(test)
    print(train_point)
    print(test_point)

def stat_real_disasters(data):
    true = 0
    false = 0
    for i in data:
        if i[4] == '1':
            true += 1
        else:
            false += 1
    print(true)
    print(false)

def split_train(data):
    ratio = 0.7
    train_set = []
    len_train = 0
    len_total = len(data)
    indexes = []
    removed_set = set()
    while len_train < int(0.7*len_total):
        index = int(random.random() * len(data))
        while (index in removed_set):
            index = int(random.random() * len(data))
        removed_set.add(index)
        indexes.append(data[index])
        len_train += 1
    for i in indexes:
        train_set.append(i)
        data.remove(i)
    dev_set = np.array(data)
    train_set = np.array(train_set)
    dev = {"id":dev_set[:,0].tolist(),
            "keyword":dev_set[:,1].tolist(),
            "location":dev_set[:,2].tolist(),
            "text":dev_set[:,3].tolist(),
            "target":dev_set[:,4].tolist()}
    train = {"id":train_set[:,0].tolist(),
            "keyword":train_set[:,1].tolist(),
            "location":train_set[:,2].tolist(),
            "text":train_set[:,3].tolist(),
            "target":train_set[:,4].tolist()}
    path = "./nlp-getting-started/"
    t = DataFrame(train)
    d = DataFrame(dev)
    d.to_csv(path + "dev_set.csv", index=False)
    t.to_csv(path + "train_set.csv", index=False)


def preprocess_data(data):
    lemmatizer = WordNetLemmatizer()
    tokenizer = RegexpTokenizer(r'\w+')
    punc = string.punctuation
    stop_words = set(stopwords.words('english')) 
    for i in range(len(data)):
        data[i][3] = data[i][3].lower()
        # data[i][3] = re.split(' ', data[i][3])
        # data[i][3] = tokenizer.tokenize(data[i][3])
        data[i][3] = word_tokenize(data[i][3])
        delete_list = []
        for j, word in enumerate(data[i][3]):
            if word.startswith("//t"):
                delete_list.append(word)
            elif word.startswith("http"):
                delete_list.append(word)
            elif (word in punc):
                delete_list.append(word)
            elif (word.isalpha() == False):
                delete_list.append(word)
            elif (word in stop_words):
                delete_list.append(word)
        for word in delete_list:
            data[i][3].remove(word)
        for j, word in enumerate(data[i][3]):
            processed_word = lemmatizer.lemmatize(word, pos="v")
            processed_word = lemmatizer.lemmatize(processed_word, pos="a")
            processed_word = lemmatizer.lemmatize(processed_word, pos="n")
            data[i][3][j] = processed_word
    return data

def word_bags(data, mindf):
    # decide the threshold M
    words = []
    words_set = set()
    label = []
    for i in range(len(data)):
        temp = ""
        for j in range(len(data[i][3])):
            temp += data[i][3][j]
            words_set.add(data[i][3][j])
            if j!= len(data[i][3])-1:
                temp += " "
        words.append(temp)
        label.append(int(data[i][4]))
    count_vect = CountVectorizer(binary=True, min_df=mindf)
    X_train = count_vect.fit_transform(words)
    print(X_train.toarray().shape[1])
    label = np.array(label)
    return count_vect, X_train, label

def merge_data(data):
    label = []
    words = []
    for i in range(len(data)):
        temp = ""
        for j in range(len(data[i][3])):
            if j!=len(data[i][3])-1:
                temp = temp + data[i][3][j] + " "
            else:
                temp = temp + data[i][3][j]
        words.append(temp)
        label.append(int(data[i][4]))
    return words, label

def naive_bayes(data, train, dev):
    # decide the threshold M
    data = train
    count_vect, X_train, label = word_bags(data, 3)
    n = X_train.shape[0]
    d = X_train.shape[1]
    K = 2
    # shapes of parameters
    psis = np.zeros([K, d])
    phis = np.zeros([K])
    # compute the parameters
    for k in range(K):
        X_k = X_train[label == k]
        psis[k] = np.mean(X_k, axis=0)
        phis[k] = X_k.shape[0] / float(n)
    dev_data, dev_label = merge_data(dev)
    dev_data = count_vect.transform(dev_data).toarray()
    def nb_predictions(data, psis, phis):
        x = data
        n, d = x.shape
        x = np.reshape(x, (1, n, d))
        psis = np.reshape(psis, (K, 1, d))
        # clip probabilities to avoid log(0)
        psis = psis.clip(1e-14, 1-1e-14)
        # compute log-probabilities
        logpy = np.log(phis).reshape([K,1])
        logpxy = x * np.log(psis) + (1-x) * np.log(1-psis)
        logpyx = logpxy.sum(axis=2) + logpy
        return logpyx.argmax(axis=0).flatten(), logpyx.reshape([K,n])
    idx, logpyx = nb_predictions(dev_data, psis, phis)
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    for i, j in zip(idx, dev_label):
        if i and j:
            TP += 1
        elif i==1 and j!=1:
            FP += 1
        elif i!=1 and j==1:
            FN += 1
        else:
            TN += 1
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1 = 2*(precision * recall) / (precision + recall)
    print(F1)
    return F1

def get_weight(count_vect, model, num):
    # get important weight
    coef = model.coef_
    coef = [abs(i) for i in coef[0]]
    max_num_index_list = map(coef.index, heapq.nlargest(num, coef))
    max_num_index_list = list(max_num_index_list)
    import_words = set()
    most_ = max_num_index_list
    for i in count_vect.vocabulary_.keys():
        if count_vect.vocabulary_[i] in most_:
            import_words.add(i)
    influential_words = []
    for i in import_words:
        influential_words.append(i)
    return influential_words

def logistic(data, train, dev):
    # decide the threshold M
    data = train
    count_vect, X_train, label = word_bags(data, 1)
    dev_data, dev_label = merge_data(dev)
    dev_data = count_vect.transform(dev_data).toarray()
    logreg = LogisticRegression(C=1e5, multi_class='multinomial', verbose=True)
    logreg.fit(X_train, label)
    dev_predict = logreg.predict(dev_data)
    F1 = f1_score(dev_label, dev_predict)
    print(F1)
    # get important weight
    influential_words = get_weight(count_vect, logreg, 10)
    print(influential_words)
    return influential_words, F1

def linear_svm(data, train, dev, c):
    # decide the threshold M
    data = train
    count_vect, X_train, label = word_bags(data, 1)
    dev_data, dev_label = merge_data(dev)
    dev_data = count_vect.transform(dev_data).toarray()
    clf = LinearSVC(penalty='l2', loss='hinge', dual=True, C=c)
    clf.fit(X_train, label)
    dev_predict = clf.predict(dev_data)
    F1 = f1_score(dev_label, dev_predict)
    print(F1)
    # get important weight
    influential_words = get_weight(count_vect, clf, 10)
    print(influential_words)
    return influential_words, F1

def C_plot(total_processed_data, processed_data, processed_dev, C):
    F = []
    for i in C:
        a, b = linear_svm(total_processed_data, processed_data, processed_dev, i)
        F.append(b)
    print(F)
    plt.plot([0.01, 0.1, 1, 10, 100], F)
    plt.xticks([0.01, 0.1, 1, 10, 100])
    plt.savefig("./linear_svm.png")


def non_linear_svm(data, train, dev, c):
    # decide the threshold M
    data = train
    count_vect, X_train, label = word_bags(data, 1)
    dev_data, dev_label = merge_data(dev)
    dev_data = count_vect.transform(dev_data).toarray()
    clf = SVC(C=c, kernel="rbf")
    clf.fit(X_train, label)
    dev_predict = clf.predict(dev_data)
    F1 = f1_score(dev_label, dev_predict)
    print(F1)
    return F1

def non_lin_C_plot(total_processed_data, processed_data, processed_dev, C):
    F = []
    for i in C:
        b = non_linear_svm(total_processed_data, processed_data, processed_dev, i)
        F.append(b)
    print(F)
    plt.scatter([0.01, 0.1, 1, 10, 100], F)
    plt.xticks([0.01, 0.1, 1, 10, 100])
    plt.savefig("./non_linear_svm.png")

def n_gram(data, mindf):
    # decide the threshold M
    words = []
    words_set = set()
    label = []
    for i in range(len(data)):
        temp = ""
        for j in range(len(data[i][3])):
            temp += data[i][3][j]
            words_set.add(data[i][3][j])
            if j!= len(data[i][3])-1:
                temp += " "
        words.append(temp)
        label.append(int(data[i][4]))
    count_vect = CountVectorizer(binary=True, min_df=mindf, ngram_range=(1,2))
    X_train = count_vect.fit_transform(words)

    X_show = X_train.toarray()
    # print(X_show.shape[0])
    # print(X_show.shape[1])
    # print(len(count_vect.vocabulary_))

    # two_gram_vec = CountVectorizer(binary=True, min_df=mindf, ngram_range=(2,2))
    # two_gram_vec.fit_transform(words)
    # countt = 0
    # # print(type(two_gram_vec.vocabulary_))
    # for i in two_gram_vec.vocabulary_.keys():
    #     print(i)
    #     countt += 1
    #     if countt == 9:
    #         break
    label = np.array(label)
    return count_vect, X_train, label

def n_gram_naive_bayes(data, train, dev):
    # decide the threshold M
    data = train
    count_vect, X_train, label = n_gram(data, 3)
    n = X_train.shape[0]
    d = X_train.shape[1]
    K = 2
    # shapes of parameters
    psis = np.zeros([K, d])
    phis = np.zeros([K])
    # compute the parameters
    for k in range(K):
        X_k = X_train[label == k]
        psis[k] = np.mean(X_k, axis=0)
        phis[k] = X_k.shape[0] / float(n)
    dev_data, dev_label = merge_data(dev)
    dev_data = count_vect.transform(dev_data).toarray()
    def nb_predictions(data, label, psis, phis):
        label = label
        x = data
        n, d = x.shape
        x = np.reshape(x, (1, n, d))
        psis = np.reshape(psis, (K, 1, d))
        # clip probabilities to avoid log(0)
        psis = psis.clip(1e-14, 1-1e-14)
        # compute log-probabilities
        logpy = np.log(phis).reshape([K,1])
        logpxy = x * np.log(psis) + (1-x) * np.log(1-psis)
        logpyx = logpxy.sum(axis=2) + logpy
        return logpyx.argmax(axis=0).flatten(), logpyx.reshape([K,n]), label
    idx, logpyx, dev_label = nb_predictions(dev_data, dev_label, psis, phis)
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    for i, j in zip(idx, dev_label):
        if i and j:
            TP += 1
        elif i==1 and j!=1:
            FP += 1
        elif i!=1 and j==1:
            FN += 1
        else:
            TN += 1
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1 = 2*(precision * recall) / (precision + recall)
    print(F1)
    return F1

def n_gram_logistic(data, train, dev):
    # decide the threshold M
    data = train
    count_vect, X_train, label = n_gram(data, 1)
    dev_data, dev_label = merge_data(dev)
    dev_data = count_vect.transform(dev_data).toarray()
    logreg = LogisticRegression(C=1e5, multi_class='multinomial', verbose=True)
    logreg.fit(X_train, label)
    dev_predict = logreg.predict(dev_data)
    F1 = f1_score(dev_label, dev_predict)
    print(F1)
    # get important weight
    influential_words = get_weight(count_vect, logreg, 10)
    print(influential_words)
    return influential_words, F1

def n_gram_linear_svm(data, train, dev, c):
    # decide the threshold M
    data = train
    count_vect, X_train, label = n_gram(data, 1)
    dev_data, dev_label = merge_data(dev)
    dev_data = count_vect.transform(dev_data).toarray()
    clf = LinearSVC(penalty='l2', loss='hinge', dual=True, C=c)
    clf.fit(X_train, label)
    dev_predict = clf.predict(dev_data)
    F1 = f1_score(dev_label, dev_predict)
    print(F1)
    # get important weight
    influential_words = get_weight(count_vect, clf, 10)
    print(influential_words)
    return influential_words, F1

def n_gram_non_linear_svm(data, train, dev, c):
    # decide the threshold M
    data = train
    count_vect, X_train, label = n_gram(data, 1)
    dev_data, dev_label = merge_data(dev)
    dev_data = count_vect.transform(dev_data).toarray()
    clf = SVC(C=c, kernel="rbf")
    clf.fit(X_train, label)
    dev_predict = clf.predict(dev_data)
    F1 = f1_score(dev_label, dev_predict)
    print(F1)
    return F1

def ngram_C_plot(total_processed_data, processed_data, processed_dev, C):
    F = []
    for i in C:
        a, b = n_gram_linear_svm(total_processed_data, processed_data, processed_dev, i)
        F.append(b)
    print(F)
    plt.plot([0.01, 0.1, 1, 10, 100], F)
    plt.xticks([0.01, 0.1, 1, 10, 100])
    plt.savefig("./n_gram_linear_svm.png")

def ngram_non_lin_C_plot(total_processed_data, processed_data, processed_dev, C):
    F = []
    for i in C:
        b = n_gram_non_linear_svm(total_processed_data, processed_data, processed_dev, i)
        F.append(b)
    print(F)
    plt.scatter([0.01, 0.1, 1, 10, 100], F)
    plt.xticks([0.01, 0.1, 1, 10, 100])
    plt.savefig("./n_gram_non_linear_svm.png")

def append_data(data):
    for i in range(len(data)):
        temp_str = ""
        temp_str  = temp_str + data[i][1] + " "
        temp_str  = temp_str + data[i][2] + " "
        temp_str  = temp_str + data[i][3]
        data[i][3] = temp_str
    return data

def trans(data, count_vect):
    words = []
    for i in range(len(data)):
        temp = ""
        for j in range(len(data[i][3])):
            if j!=len(data[i][3])-1:
                temp = temp + data[i][3][j] + " "
            else:
                temp = temp + data[i][3][j]
        words.append(temp)
    return words


def predict_test(train, dev):
    data = train
    id = []
    for i in range(len(dev)):
        id.append(dev[i][0])
    count_vect, X_train, label = word_bags(data, 3)
    n = X_train.shape[0]
    d = X_train.shape[1]
    K = 2
    # shapes of parameters
    psis = np.zeros([K, d])
    phis = np.zeros([K])
    # compute the parameters
    for k in range(K):
        X_k = X_train[label == k]
        psis[k] = np.mean(X_k, axis=0)
        phis[k] = X_k.shape[0] / float(n)
    dev_data = trans(dev, count_vect)
    dev_data = count_vect.transform(dev_data).toarray()
    def nb_predictions(data, psis, phis):
        x = data
        n, d = x.shape
        x = np.reshape(x, (1, n, d))
        psis = np.reshape(psis, (K, 1, d))
        # clip probabilities to avoid log(0)
        psis = psis.clip(1e-14, 1-1e-14)
        # compute log-probabilities
        logpy = np.log(phis).reshape([K,1])
        logpxy = x * np.log(psis) + (1-x) * np.log(1-psis)
        logpyx = logpxy.sum(axis=2) + logpy
        return logpyx.argmax(axis=0).flatten(), logpyx.reshape([K,n])
    idx, logpyx = nb_predictions(dev_data, psis, phis)
    ex = {"id":id,
            "target":idx}
    d=DataFrame(ex)
    d.to_csv("./submit.csv", index=False)

if __name__ == "__main__":
    path = "./nlp-getting-started/train.csv"
    test = "./nlp-getting-started/test.csv"
    train_set = "./nlp-getting-started/train_set.csv"
    dev_set = "./nlp-getting-started/dev_set.csv"
    data = read_data(path)
    test_data = read_data(test)
    # stat_train_test_data_point(data, test_data)
    # stat_real_disasters(data)
    # split_train(data)
    train_set = read_data(train_set)
    dev_set = read_data(dev_set)
    total_processed_data = preprocess_data(data)
    processed_data = preprocess_data(train_set)
    processed_dev = preprocess_data(dev_set)
    # word_bags(processed_data, 1)
    # n_gram(processed_data, 1)
    # naive_bayes_F1 = naive_bayes(total_processed_data, processed_data, processed_dev)
    # important_words, log_F1 = logistic(total_processed_data, processed_data, processed_dev)
    # C_plot(total_processed_data, processed_data, processed_dev, [0.01, 0.1, 1, 10, 100])
    # linear_svm_words, linear_svm_F1 = linear_svm(total_processed_data, processed_data, processed_dev, 0.1)
    # non_linear_svm_F1 = non_linear_svm(total_processed_data, processed_data, processed_dev, 0.1)
    # non_lin_C_plot(total_processed_data, processed_data, processed_dev, [0.01, 0.1, 1, 10, 100])
    n_gram_naive_bayes_F1 = n_gram_naive_bayes(total_processed_data, processed_data, processed_dev)
    # n_gram_important_words, n_gram_log_F1 = n_gram_logistic(total_processed_data, processed_data, processed_dev)
    n_gram_linear_svm_words, n_gram_linear_svm_F1 = n_gram_linear_svm(total_processed_data, processed_data, processed_dev, 0.1)
    n_gram_non_linear_svm_F1 = n_gram_non_linear_svm(total_processed_data, processed_data, processed_dev, 10)
    # ngram_C_plot(total_processed_data, processed_data, processed_dev, [0.01, 0.1, 1, 10, 100])
    # ngram_non_lin_C_plot(total_processed_data, processed_data, processed_dev, [0.01, 0.1, 1, 10, 100])
    # append_total_data = append_data(data)
    # append_train_set = append_data(train_set)
    # append_dev_set = append_data(dev_set)
    # total_processed_data = preprocess_data(append_total_data)
    # processed_data = preprocess_data(append_train_set)
    # processed_dev = preprocess_data(append_dev_set)
    # naive_bayes_F1 = naive_bayes(total_processed_data, processed_data, processed_dev)
    # important_words, log_F1 = logistic(total_processed_data, processed_data, processed_dev)
    # C_plot(total_processed_data, processed_data, processed_dev, [0.01, 0.1, 1, 10, 100])
    # linear_svm_words, linear_svm_F1 = linear_svm(total_processed_data, processed_data, processed_dev, 0.1)
    # non_linear_svm_F1 = non_linear_svm(total_processed_data, processed_data, processed_dev)
    # non_lin_C_plot(total_processed_data, processed_data, processed_dev, [0.01, 0.1, 1, 10, 100])
    # n_gram_naive_bayes_F1 = n_gram_naive_bayes(total_processed_data, processed_data, processed_dev)
    # n_gram_important_words, n_gram_log_F1 = n_gram_logistic(total_processed_data, processed_data, processed_dev)
    # n_gram_linear_svm_words, n_gram_linear_svm_F1 = n_gram_linear_svm(total_processed_data, processed_data, processed_dev)
    # n_gram_non_linear_svm_F1 = n_gram_non_linear_svm(total_processed_data, processed_data, processed_dev)
    test_data = preprocess_data(test_data)
    predict_test(total_processed_data, test_data)