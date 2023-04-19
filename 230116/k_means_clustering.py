import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from sklearn.utils import shuffle

np.random.seed(0)

n_samples = 100
means = [[1,1], [1,-1], [-1,1], [-1,-1]]
stds = [0.5, 0.4, 0.3, 0.2]
datas = []

n_cluster = len(means)

cmap = cm.get_cmap('tab20')

fig, axes = plt.subplots(2, 3, figsize=(10,7))
ax = axes.flatten()

# random data 생성
for i in range(n_cluster):
    data = np.random.normal(loc=means[i], scale=stds[i],
                            size=(n_samples, 2))
    datas.append(data)

    ax[0].scatter(data[:, 0], data[:, 1], color=cmap(0), alpha=0.7)
    ax[0].tick_params(labelsize=10)

# random dots 생성
# 이렇게 하는거 아님..
# dots = np.random.rand(4,2)
# ax[0].scatter(dots[:,0], dots[:,1], color='r')

# random dots 생성
datas = np.vstack(datas) # (400,2)
shuffle_datas = shuffle(datas, random_state=0)
dots = shuffle_datas[:4]

ax[0].scatter(dots[:, 0], dots[:, 1], color='r', s=50)

for ax_idx in range(1,len(ax)):
    # clustering
    colors = [cmap(i) for i in range(n_cluster)]
    min_idx = [] # 거리 제일 작은 점 인덱스


    for data in datas:
        distances = np.sum((dots - data) ** 2, axis=1)
        cluster = np.argmin(distances)
        min_idx.append(cluster)


    ax[ax_idx].scatter(datas[:,0], datas[:, 1], c=cmap(min_idx), alpha=0.7)

    # cluster별 평균 구하기
    min_idx = np.array(min_idx)

    cluster_means = []
    for cluster_idx in range(n_cluster):
        data = datas[min_idx == cluster_idx]
        x_mean = np.sum(data[:,0]) / len(data) # 100 아냐...
        y_mean = np.sum(data[:,1]) / len(data)
        cluster_means.append([x_mean, y_mean])

    cluster_means = np.array(cluster_means)
    ax[ax_idx].scatter(cluster_means[:, 0], cluster_means[:, 1], color='r', s=50)

    # dots(centeroid) 좌표값 갱신
    dots = cluster_means

plt.show()