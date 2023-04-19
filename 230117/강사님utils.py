import numpy as np

def get_dataset():
    n_cluster_samples = 100
    means = [[1, 1], [-1, 1], [-1, -1], [1, -1]]
    stds = [0.5, 0.4, 0.3, 0.2]

    n_clusters = len(means)
    data = []
    for cluster_idx in range(n_clusters):
        cluster_data = np.random.normal(loc=means[cluster_idx],
                                        scale=stds[cluster_idx],
                                        size=(n_cluster_samples, 2))
        data.append(cluster_data)
    data = np.vstack(data)
    return data
# data는 400개인 상태에서.. 거기서 random centroids 개수를 정하면 그만큼 나눠주는 거임.!!
# means, stds는 그 400개를 만들기 위한 기준점일 뿐!!!!!!!

def get_initial_centroids(dataset, K):
    n_data = dataset.shape[0]

    random_idx = np.arange(n_data)
    np.random.shuffle(random_idx)
    centroid_idx = random_idx[:K] # K개!!!

    centroids = dataset[centroid_idx]
    return centroids


def cal_which_cluster(dataset, centroids, sample_idx):
    sample = dataset[sample_idx]

    dists = []
    for centroid in centroids:
        dist = np.sum((sample - centroid) ** 2)
        dists.append(dist)

    which_cluster = np.argmin(dists)
    return which_cluster


def clustering(dataset, centroids, K):
    n_data = dataset.shape[0]
    clusters = [[] for _ in range(K)]

    for sample_idx in range(n_data):
        which_cluster = cal_which_cluster(dataset, centroids, sample_idx)
        clusters[which_cluster].append(dataset[sample_idx])

    clusters = [np.array(cluster) for cluster in clusters]
    return clusters


def update_centroids(clusters):
    centroids = np.array([np.mean(cluster, axis=0)
                          for cluster in clusters])
    return centroids