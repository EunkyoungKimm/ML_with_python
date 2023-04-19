# 실행창2.py

from utils2 import get_dataset
from utils2 import get_initial_centroids
from utils2 import clustering
from utils2 import update_centroids

import matplotlib.pyplot as plt
import matplotlib.cm as cm

# get_dataset
datas = get_dataset()

cmap = cm.get_cmap('tab20')

fig, axes = plt.subplots(2, 3, figsize=(15,10))
ax = axes.flatten()

ax[0].scatter(datas[:, 0], datas[:, 1], color=cmap(0), alpha=0.7)
ax[0].tick_params(labelsize=10)

# get_initial_centroids
centroids = get_initial_centroids(datas)
ax[0].scatter(centroids[:, 0], centroids[:, 1], color='r', s=80)

# clustering and update_centroids
for ax_idx in range(1, len(ax)):
    min_idx = clustering(datas, centroids)
    ax[ax_idx].scatter(datas[:, 0], datas[:, 1], c=cmap(min_idx), alpha=0.7)

    for cluster_idx in range(len(centroids)):
        dots = update_centroids(datas, min_idx)
        ax[ax_idx].scatter(centroids[:, 0], centroids[:, 1], color='r', s=50)

plt.show()




