# 함수화!
# UnboundLocalError: local variable 'datas' referenced before assignment
# 지역변수.. 어떤 함수의 return값을 다른 함수에 사용하려면 어떻게 할지.. 이것때문에 많이 애먹음!
# unpacking >> https://codechacha.com/ko/python-return-multiple-values/
# (어떤 함수)return datas, n_cluster, cmap, ax > (다른 함수)datas, n_cluster, cmap, ax = get_dataset()
# plt.show()는 한 군데에서만!!! 아니면 retrun을 plt.show위에서 해주면 됨!

# scatter하는 부분은 따로 코드로 작성해서 실행하셨다고 함..흑..ㅜㅜㅜ함수로 구현안하고!! 이렇게 해보장!!

# utils.py

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import random

from sklearn.utils import shuffle

# get_dataset
def get_dataset():
    k = int(input('원하는 k값을 입력하세요: '))

    np.random.seed(0)
    n_samples = 100
    means = []
    stds = []

    # means, stds
    for i in range(k):
        mean = []
        for j in range(2):
            m = random.uniform(-1, 1)
            mean.append(m)

        std = random.uniform(0, 1)

        means.append(mean)
        stds.append(std)

    datas = []
    n_cluster = k
    cmap = cm.get_cmap('tab20')

    fig, axes = plt.subplots(2, 3, figsize=(15,10))
    ax = axes.flatten()

    # random data 생성
    for i in range(n_cluster):
        data = np.random.normal(loc=means[i], scale=stds[i],
                                size=(n_samples, 2))
        datas.append(data)

        ax[0].scatter(data[:, 0], data[:, 1], color=cmap(0), alpha=0.7)
        ax[0].tick_params(labelsize=10)


    return datas, n_cluster, cmap, ax


# get_initial_centroids
def get_initial_centroids(): # 여기는 parameter!  입력해줘야 됨
    datas, n_cluster, cmap, ax = get_dataset()
    # random dots 생성
    datas = np.vstack(datas)
    shuffle_datas = shuffle(datas, random_state=0)
    dots = shuffle_datas[:n_cluster]

    ax[0].scatter(dots[:, 0], dots[:, 1], color='r', s=80)

    return dots, datas, n_cluster, cmap, ax

# clustering and update centroids
def clustering_and_update_centroids():
    dots, datas, n_cluster, cmap, ax = get_initial_centroids()

    # clustering
    for ax_idx in range(1, len(ax)):
        min_idx = []  # 거리 제일 작은 점 인덱스

        for data in datas:
            distances = np.sum((dots - data) ** 2, axis=1)
            cluster = np.argmin(distances)
            min_idx.append(cluster)

        ax[ax_idx].scatter(datas[:, 0], datas[:, 1], c=cmap(min_idx), alpha=0.7)

        # cluster별 평균 구하기
        min_idx = np.array(min_idx)

        cluster_means = []
        for cluster_idx in range(n_cluster):
            data = datas[min_idx == cluster_idx]
            print(data)
            x_mean = np.sum(data[:, 0]) / len(data)  # 100 아냐...
            y_mean = np.sum(data[:, 1]) / len(data)
            cluster_means.append([x_mean, y_mean])

        cluster_means = np.array(cluster_means)
        ax[ax_idx].scatter(cluster_means[:, 0], cluster_means[:, 1], color='r', s=50)

        # update centeroid
        dots = cluster_means

    plt.show()