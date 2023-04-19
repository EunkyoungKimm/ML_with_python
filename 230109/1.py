import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm

np.random.seed(4)

n_class = 20
n_data = 100
center_pt = np.random.uniform(-60, 60, (n_class, 2))
# print(center_pt)
cmap = cm.get_cmap('tab20')
colors = [cmap(i) for i in range(n_class)]

data_list = []
for class_idx in range(n_class):
    center = center_pt[class_idx]

    x_data = center[0] + 2*np.random.normal(0, 1, (1, n_data))
    y_data = center[1] + 2*np.random.normal(0, 1, (1, n_data))
    data = np.vstack((x_data, y_data))

    data_list.append(data)

# print(data_list)

fig, ax = plt.subplots(figsize=(10,10))
for class_idx in range(n_class):
    data = data_list[class_idx]
    ax.scatter(data[0], data[1],
               s=30,
               facecolor=colors[class_idx])
fig.tight_layout()
plt.show()