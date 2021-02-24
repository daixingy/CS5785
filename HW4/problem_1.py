import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn import cluster


def distance(data, centroids):
    dist = []
    for i in range(len(centroids)):
        a = np.array(centroids[i])
        b = np.array(data)
        dist.append(np.linalg.norm(a - b))
    return np.array(dist)

class KMeans():
    def __init__(self, k=2, max_iter=100, epsilon=0.00001):
        self.k = k
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.cluster_centers_ = None
        self.labels_ = None
    
    def random_centroids(self, data):
        num_samples, num_features = np.shape(data)
        centroids = np.zeros((self.k, num_features))
        for i in range(self.k):
            centroid = data[np.random.choice(range(num_samples))]
            centroids[i] = centroid
        return centroids
    
    def get_label(self, data, centroids):
        distances = distance(data, centroids)
        label = np.argmin(distances)
        return label

    def get_clusters(self, centroids, data):
        num_samples = np.array(data).shape[0]
        clusters = []
        for i in range(self.k):
            clusters.append([])
        for index, sample in enumerate(data):
            centroid_index = self.get_label(sample, centroids)
            clusters[centroid_index].append(index)
        return clusters
    
    def update_centroids(self, clusters, data):
        num_features = np.array(data).shape[1]
        centroids = np.zeros((self.k, num_features))
        for index, data_index in enumerate(clusters):
            centroid = np.mean(np.array(data)[data_index], axis=0)
            centroids[index] = centroid
        self.cluster_centers_ = centroids
        return centroids

    def get_cluster_labels(self, clusters, data):
        y_pred = np.zeros(np.array(data).shape[0])
        for index, data_index in enumerate(clusters):
            for sample in data_index:
                y_pred[sample] = index
        self.labels_ = y_pred
        return y_pred
    
    def fit(self, data):
        centroids = self.random_centroids(data)
        
        for i in range(self.max_iter):
            
            clusters = self.get_clusters(centroids, data)
            previous_centroids = centroids

            centroids = self.update_centroids(clusters, data)
            diff = centroids - previous_centroids
            if diff.any() < self.epsilon:
                break
        self.get_cluster_labels(clusters, data)

    def score(self, data):
        self.fit(data)
        score = 0
        for i in range(len(data)):
            score = score + np.linalg.norm(np.array(data[i]) - np.array(self.cluster_centers_[int(self.labels_[i])]))
        return score


def problem_1_a(path_1, path_2, path_3):
    word = np.load(path_1)
    # print(word[0])
    print(word.shape)
    k = 2
    objs = []
    Ks = []
    while (k<=25):
        print(k)
        model = KMeans(k=k)
        model.fit(word)
        objs.append(model.score(word))
        Ks.append(k)
        k += 1
    plt.plot(Ks, objs, '.-', markersize=15)
    plt.xlabel("Number of clusters K")
    plt.ylabel("Objective Function Value")
    plt.savefig("./444.png")

    # k = 17
    # # get mean
    # model = cluster.KMeans(n_clusters=k)
    # model.fit(word)
    # X_mean = np.mean(word, axis=0)
    # print(X_mean)
    # X_i = model.cluster_centers_
    # print(X_i.shape)
    # f = open(path_2)
    # lines = []
    # line = f.readline()
    # while line:
    #     line = line[:-1]
    #     lines.append(line)
    #     line = f.readline()
    # f.close()
    # # print(lines)
    # vocab = lines
    # top_components = []
    # # report the words associated with the top components
    # components_index = []
    # for i in range(len(X_i)):
    #     distance = abs(X_i[i] - X_mean)
    #     indexes = distance.argsort()[-10:][::-1]
    #     components_index.append(indexes)
    #     top_components.append(np.array(vocab)[indexes].tolist())
    # print(top_components)

    # # report ten documents that have samllest distance to its mean vector
    # print(model.labels_)
    # ten_documents = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
    # ten_documents_index = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
    # for i in range(len(word)):
    #     label = model.labels_[i]
    #     cluster_mean = X_i[label]
    #     distance = np.linalg.norm(word[i] - cluster_mean)
    #     ten_documents[label].append(distance)
    #     ten_documents_index[label].append(i)
    
    # for i in range(len(ten_documents)):
    #     ten_documents[i] = np.argsort(np.array(ten_documents[i]))[:10]
    
    # for i in range(len(ten_documents_index)):
    #     ten_documents_index[i] = np.array(ten_documents_index[i])[ten_documents[i]]
    
    # print(ten_documents_index)

    # f = open(path_3)
    # lines = []
    # line = f.readline()
    # while line:
    #     line = line[:-1]
    #     lines.append(line)
    #     line = f.readline()
    # f.close()
    # titles = lines
    # title_clsuters = []
    # for i in range(len(ten_documents_index)):
    #     title_clsuters.append(np.array(titles)[ten_documents_index[i].tolist()].tolist())
    
    # for i in range(len(title_clsuters)):
    #     print(title_clsuters[i])
    #     print()

def problem_1_b(path_1, path_2, path_3):
    word = np.load(path_1)
    # print(word[0])
    print(word.shape)
    k = 2
    objs = []
    Ks = []
    while (k<=25):
        print(k)
        model = KMeans(k=k)
        model.fit(word)
        objs.append(model.score(word))
        Ks.append(k)
        k += 1
    plt.plot(Ks, objs, '.-', markersize=15)
    plt.xlabel("Number of clusters K")
    plt.ylabel("Objective Function Value")
    plt.savefig("./555.png")

    # k = 6
    # # get mean
    # model = cluster.KMeans(n_clusters=k)
    # model.fit(word)
    # X_mean = np.mean(word, axis=0)
    # print(X_mean)
    # X_i = model.cluster_centers_
    # print(X_i.shape)
    # f = open(path_2)
    # lines = []
    # line = f.readline()
    # while line:
    #     line = line[:-1]
    #     lines.append(line)
    #     line = f.readline()
    # f.close()
    # # print(lines)
    # vocab = lines
    # top_components = []
    # # report the words associated with the top components
    # components_index = []
    # for i in range(len(X_i)):
    #     distance = abs(X_i[i] - X_mean)
    #     indexes = distance.argsort()[-10:][::-1]
    #     components_index.append(indexes)
    #     top_components.append(np.array(vocab)[indexes].tolist())
    # for i in range(len(top_components)):
    #     print(top_components[i])
    #     print()

    # # report ten documents that have samllest distance to its mean vector
    # print(model.labels_)
    # # ten_documents = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
    # # ten_documents_index = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
    # ten_documents = [[],[],[],[],[],[]]
    # ten_documents_index = [[],[],[],[],[],[]]
    # for i in range(len(word)):
    #     label = model.labels_[i]
    #     cluster_mean = X_i[label]
    #     distance = np.linalg.norm(word[i] - cluster_mean)
    #     ten_documents[label].append(distance)
    #     ten_documents_index[label].append(i)
    
    # for i in range(len(ten_documents)):
    #     ten_documents[i] = np.argsort(np.array(ten_documents[i]))[:10]
    
    # for i in range(len(ten_documents_index)):
    #     ten_documents_index[i] = np.array(ten_documents_index[i])[ten_documents[i]]
    
    # print(ten_documents_index)

    # f = open(path_3)
    # lines = []
    # line = f.readline()
    # while line:
    #     line = line[:-1]
    #     lines.append(line)
    #     line = f.readline()
    # f.close()
    # titles = lines
    # title_clsuters = []
    # for i in range(len(ten_documents_index)):
    #     title_clsuters.append(np.array(titles)[ten_documents_index[i].tolist()].tolist())
    
    # for i in range(len(title_clsuters)):
    #     print(title_clsuters[i])
    #     print()





if __name__ == "__main__":
    data = "./data/"
    file_1 = "science2k-vocab.txt"
    file_2 = "science2k-titles.txt"
    file_3 = "science2k-doc-word.npy"
    file_4 = "science2k-word-doc.npy"
    # print(data)
    problem_1_a(data + file_3, data + file_1, data + file_2)
    # problem_1_b(data + file_4, data + file_2, data + file_1)


    