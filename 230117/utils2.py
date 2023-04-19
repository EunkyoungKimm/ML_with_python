# scatter 부분은 실행창에서!
#
#단계별로 return값은 다음 함수에 필요한 하나 정도로만 받아서 작성하고,
# scatter는 그냥 아예 뒷부분에 따로 빼버리면
# 생각보다 코드가 간단해질 수 있어요

# utils2.py

import numpy as np
import random
from sklearn.utils import shuffle


k = int(input('원하는 k값을 입력하세요: ')) # 이렇게 하면 모든 함수에서 k 사용 가능!
n_cluster = k

# get_dataset
def get_dataset():
    random.seed(0)

    n_samples = 100
    means = []
    stds = []

    # k만큼 means,stds 생성
    for i in range(n_cluster):
        mean = []
        for j in range(2):
            m = random.uniform(-1, 1)
            mean.append(m)

        std = random.uniform(0, 1)

        means.append(mean)
        stds.append(std)

    datas = []
    # random data 생성
    for i in range(n_cluster):
        data = np.random.normal(loc=means[i], scale=stds[i],
                                size=(n_samples, 2))
        datas.append(data)

    datas = np.vstack(datas) # list > numpy.ndarray

    return datas

# get_initial_centroids
def get_initial_centroids(datas):
    shuffle_datas = shuffle(datas, random_state=0)
    centroids = shuffle_datas[:n_cluster]

    return centroids

# clustering0
def clustering(datas, centroids):
    min_idx = []  # 거리 제일 작은 점 인덱스

    for data in datas:
        distances = np.sum((centroids - data) ** 2, axis=1)
        cluster = np.argmin(distances)
        min_idx.append(cluster)
    min_idx = np.array(min_idx)

    return min_idx

# update_centroids
def update_centroids(datas, min_idx):
    cluster_means = []
    for cluster_idx in range(n_cluster):
        data = datas[min_idx == cluster_idx]
        x_mean = np.sum((data[:, 0]) / len(data))
        y_mean = np.sum((data[:, 1]) / len(data))
        cluster_means.append([x_mean, y_mean])

    cluster_means = np.array(cluster_means)

    centroids = cluster_means

    return centroids


