import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn import cluster
from sklearn import mixture
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal
# plt.style.use('seaborn')
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


def problem_2_a(path):
    # clean data
    f = open(path)
    lines = []
    line = f.readline()
    while line:
        line = line[:-1]
        lines.append(line)
        line = f.readline()
    f.close()
    lines = lines[26:]
    # print(lines)

    data = []
    for i in range(len(lines)):
        temp = lines[i].split()
        temp_data = []
        temp_data.append(float(temp[1]))
        temp_data.append(float(temp[2]))
        data.append(temp_data)
    # complete clean
    print(data)
    x_axis = []
    y_axis = []
    for i in range(len(data)):
        x_axis.append(data[i][0])
        y_axis.append(data[i][1])
    plt.scatter(x_axis, y_axis)
    plt.xlabel("eruption time")
    plt.ylabel("waiting time")
    plt.savefig("./7.png")


def update_z(data, mean, sigma, ni):
    num_data, num_clusters = len(data), len(ni)
    pdfs = np.zeros((num_data, num_clusters))
    for i in range(num_clusters):
        # update probabilities conditioned on the ratio of each cluster
        pdfs[:, i] = ni[i] * multivariate_normal.pdf(data, mean[i], np.diag(sigma[i]))
    # update z matrix
    z = pdfs / pdfs.sum(axis=1).reshape(-1, 1)
    return z

def update_ni(z):
    # update pi
    ni = z.sum(axis=0) / z.sum()
    return ni

def log_likelyhood(data, ni, mean, sigma):
    num_data, num_clusters = len(data), len(ni)
    pdfs = np.zeros(((num_data, num_clusters)))
    # calculate lly value
    for i in range(num_clusters):
        pdfs[:, i] = ni[i] * multivariate_normal.pdf(data, mean[i], np.diag(sigma[i]))
    return np.mean(np.log(pdfs.sum(axis=1)))


# update mean
def update_mean(data, z):
    num_cluster = z.shape[1]
    mean = np.zeros((num_cluster, 2))
    n_i = z.sum(axis=0) / z.sum()
    for i in range(num_cluster):
        # update mean
        temp = 0
        for j in range(len(data)):
            temp = temp + np.array(data[j]) * z[j][i]
        mean[i] = temp / np.sum(np.array(z[:,i]))
    return mean

# update sigma
def update_sigma(data, mean, z):
    num_cluster = z.shape[1]
    sigma = np.zeros((num_cluster, 2))
    n_i = z.sum(axis=0) / z.sum()
    for i in range(num_cluster):
        # update sigma
        # temp = 0
        # for j in range(len(data)):
        #     t = np.array(data[j]) - mean[i]
        #     t = t ** 2
        #     t = t * z[j][i]
        #     temp = temp + t
        # sigma[i] = temp / np.sum(np.array(z[:,i]))
        sigma[i] = np.average((data - mean[i]) ** 2, axis=0, weights=z[:, i])
    return sigma

def problem_2_b(path):
    # clean data
    f = open(path)
    lines = []
    line = f.readline()
    while line:
        line = line[:-1]
        lines.append(line)
        line = f.readline()
    f.close()
    lines = lines[26:]
    # print(lines)

    data = []
    for i in range(len(lines)):
        temp = lines[i].split()
        temp_data = []
        temp_data.append(float(temp[1]))
        temp_data.append(float(temp[2]))
        data.append(temp_data)
    # complete clean
    # print(data)


    # EM algorithm to estimate the parameters
    num_cluster = 2
    num_data = len(data)
    # mean = [[1,45], [3,70]]
    mean = [[np.random.rand()*2, np.random.rand()*100],[np.random.rand()*6, np.random.rand()*140]]
    sigma = [[1,2], [3,4]]
    ni = [1 / num_cluster] * 2
    z = np.ones((num_data, num_cluster)) / num_cluster
    ni = z.sum(axis = 0) / z.sum()

    # iteration
    llh = []
    means = []
    mean_difference = [[10,10],[10,10]]
    iteration_count = 0
    while(mean_difference[0][0] > 0.1 or mean_difference[0][1] > 0.1 or mean_difference[1][0] > 0.1 or mean_difference[1][1] > 0.1):
        llh.append(log_likelyhood(data, ni, mean, sigma))
        z = update_z(data, mean, sigma, ni)
        ni = update_ni(z)
        previous_mean = mean
        mean = update_mean(data, z)
        mean_difference = abs(mean - previous_mean)
        means.append(mean[0])
        means.append(mean[1])
        sigma = update_sigma(data, mean, z)
        iteration_count += 1
    
    print("iteration_number:", iteration_count)
    x_axis_1 = []
    y_axis_1 = []
    x_axis_2 = []
    y_axis_2 = []
    for i in range(0, len(means) - 1, 2):
        x_axis_1.append(means[i][0])
        y_axis_1.append(means[i][1])
        x_axis_2.append(means[i+1][0])
        y_axis_2.append(means[i+1][1])
    
    print("EM result:")
    print(mean)
    print(sigma)
    x_axis = []
    y_axis = []
    for i in range(len(data)):
        x_axis.append(data[i][0])
        y_axis.append(data[i][1])
    plt.scatter(x_axis, y_axis)
    
    plt.scatter(x_axis_1, y_axis_1, color = "r")
    plt.scatter(x_axis_2, y_axis_2, color = "g")
    plt.xlabel("eruption time")
    plt.ylabel("waiting time")
    plt.savefig("./mean.png")


def problem_2_b_2(path):
    # clean data
    f = open(path)
    lines = []
    line = f.readline()
    while line:
        line = line[:-1]
        lines.append(line)
        line = f.readline()
    f.close()
    lines = lines[26:]
    # print(lines)

    data = []
    for i in range(len(lines)):
        temp = lines[i].split()
        temp_data = []
        temp_data.append(float(temp[1]))
        temp_data.append(float(temp[2]))
        data.append(temp_data)
    # complete clean
    # print(data)


    # EM algorithm to estimate the parameters
    experiment_num = 1
    while(experiment_num <= 50):
        experiment_num += 1
        num_cluster = 2
        num_data = len(data)
        mean = [[1,45], [3,70]]
        mean = [[np.random.rand()*2, np.random.rand()*100],[np.random.rand()*6, np.random.rand()*140]]
        # mean = np.random.randint(0,100,size=(2,2))
        print("start mean:", mean)
        sigma = [[1,2], [3,4]]
        ni = [1 / num_cluster] * 2
        z = np.ones((num_data, num_cluster)) / num_cluster
        ni = z.sum(axis = 0) / z.sum()

        # iteration
        llh = []
        means = []
        mean_difference = [[10,10],[10,10]]
        iteration_count = 0
        
        while(mean_difference[0][0] > 0.1 or mean_difference[0][1] > 0.1 or mean_difference[1][0] > 0.1 or mean_difference[1][1] > 0.1):
            llh.append(log_likelyhood(data, ni, mean, sigma))
            z = update_z(data, mean, sigma, ni)
            ni = update_ni(z)
            previous_mean = mean
            mean = update_mean(data, z)
            mean_difference = abs(mean - previous_mean)
            means.append(mean[0])
            means.append(mean[1])
            sigma = update_sigma(data, mean, z)
            iteration_count += 1
        
        print("end mean:", mean)
        print("iteration_number:", iteration_count)

def problem_2_c(path):
    f = open(path)
    lines = []
    line = f.readline()
    while line:
        line = line[:-1]
        lines.append(line)
        line = f.readline()
    f.close()
    lines = lines[26:]
    # print(lines)

    data = []
    for i in range(len(lines)):
        temp = lines[i].split()
        temp_data = []
        temp_data.append(float(temp[1]))
        temp_data.append(float(temp[2]))
        data.append(temp_data)
    # complete clean
    model = KMeans(k=2)
    model.fit(data)
    X_mean = np.mean(data, axis=0)
    # print(X_mean)
    X_i = model.cluster_centers_
    # print("X_i")
    # print(X_i)
    print("K means:")
    X_mean_1 = 0
    X_mean_2 = 0
    data_1 = []
    data_2 = []
    labels = model.labels_
    for i in range(len(data)):
        if labels[i] == 0:
            X_mean_1 = X_mean_1 + np.array(data[i])
            data_1.append(data[i])
        else:
            X_mean_2 = X_mean_2 + np.array(data[i])
            data_2.append(data[i])
    X_mean_1 = np.reshape(X_mean_1, (1,2)) / len(data_1)
    X_mean_2 = np.reshape(X_mean_2, (1,2)) / len(data_2)
    counts = [len(data_1), len(data_2)]
    mean = [X_mean_1, X_mean_2]
    sigma = [[],[]]
    for i in range(2):
        # update sigma
        temp = 0
        for j in range(len(data)):
            if model.labels_[j] == i:
                t = np.array(data[j]) - mean[i]
                t = t ** 2
                temp = temp + t
        sigma[i] = temp / float(counts[i])
    print("result:")
    print(mean)
    print(sigma)
    

if __name__ == "__main__":
    path = "./data/faithful.dat"
    # problem_2_a(path)
    problem_2_b(path)
    # problem_2_b_2(path)
    problem_2_c(path)
