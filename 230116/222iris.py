import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iris = load_iris()

feature_names = iris.feature_names
n_features = len(feature_names)
species = iris.target_names
n_species = len(species)

iris_X, iris_y = iris.data, iris.target

fig, axes = plt.subplots(4, 4, figsize=(10,10))

# ax.hist
for i in range(n_features):
    feature_data = iris_X[:, i]
    axes[i][i].hist(feature_data, rwidth=0.9)

# ax.scatter
# 01 02 03 / 12 13 / 23

for dim in range(n_features-1):
    idx = [3,2,1]
    for i in idx: # 여기@@@@@@@@@@@@@@
        i = int(i)
        y_data = iris_X[:, dim]
        x_data = iris_X[:, i]
        axes[i][dim].scatter(y_data, x_data, c=iris_y, alpha=0.7)
        axes[dim][i].scatter(x_data, y_data, c=iris_y, alpha=0.7)

        axes[i][dim].grid()
        axes[dim][i].grid()
    idx.pop() # 안돼...................
    print(idx)

# set_label
for i in range(n_features):
    axes[i][0].set_ylabel(feature_names[i])
    axes[n_features-1][i].set_xlabel(feature_names[n_features-i-1])

fig.tight_layout()
plt.show()
