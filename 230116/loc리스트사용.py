import matplotlib.pyplot as plt
import numpy as np

np.random.seed(8)

n_samples = 300
x_mean, y_mean = 1, 3

data = np.random.normal(loc=[x_mean, y_mean], scale=1,
                        size=(n_samples, 2))
print(data)
print(data.shape)

fig, ax = plt.subplots(figsize=(5,5))

ax.scatter(data[:, 0], data[:, 1], alpha=0.7)

ax.scatter(x_mean, y_mean, color='r', s=100)
ax.axvline(x=x_mean, color='gray', ls=':')
ax.axhline(y=y_mean, color='gray', ls=':')

ax.tick_params(labelsize=10)
plt.show()

idx = [3,2,1]
for i in idx:
    print(i)
    print(type(i))