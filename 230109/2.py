# spine? 개념 공부하기!

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm

# data setting
names = ['DFF R-FCN', 'R-FCN', 'FGFA R-FCN']

dff_data = np.array([(0.581, 13.5),(0.598, 12.8),(0.618, 11.7),
           (0.62, 11.3), (0.624, 10.2), (0.627, 9.8),
           (0.629, 9.2), (0.63, 9)])
r_data = np.array([(0.565, 11.2), (0.645, 9)])
fgfa_data = np.array([(0.63, 8.8), (0.653, 9.3), (0.664, 9.6), (0.676, 10.1)])

dff_text = ['1:20', '1:15', '1:10', '1:8','1:5', '1:3', '1:2', '1:1']
r_text = ['Half Model', 'Full Model']
fgfa_text = ['1:1', '3:1', '7:1', '19:1']

cmap = cm.get_cmap('tab10')
colors = [cmap(i) for i in range(3)]

fig, ax = plt.subplots(figsize=(10,8))


# ax
ax.set_xlabel('mAP', fontsize=25)
ax.set_ylabel('AD', fontsize=25)
ax.tick_params(labelsize=20)

for dff_idx in range(len(dff_data)):
    ax.scatter(dff_data[dff_idx][0], dff_data[dff_idx][1],
               s=100,
               marker='o',
               facecolor=colors[0])
    plt.text(dff_data[dff_idx][0]+0.002, dff_data[dff_idx][1],
             s=dff_text[dff_idx],
             fontsize=15)

for r_idx in range(len(r_data)):
    ax.scatter(r_data[r_idx][0], r_data[r_idx][1],
               s=100,
               marker='^',
               facecolor=colors[1])
    plt.text(r_data[r_idx][0]+0.005, r_data[r_idx][1]-0.08,
             va='top', ha='center',
             s=r_text[r_idx],
             fontsize=15)

for fgfa_idx in range(len(fgfa_data)):
    ax.scatter(fgfa_data[fgfa_idx][0], fgfa_data[fgfa_idx][1],
               s=100,
               marker='*',
               facecolor=colors[2])
    plt.text(fgfa_data[fgfa_idx][0], fgfa_data[fgfa_idx][1] - 0.08,
             va='top', ha='center',
             s=fgfa_text[fgfa_idx],
             fontsize=15)

# legend
markers = ['o', '^', '*']
for i in range(len(markers)):
    ax.scatter([], [],
               s=150,
               marker=markers[i],
               facecolor=colors[i],
               label=names[i])


ax.legend(fontsize=20)

ax.grid(linestyle=':')
fig.tight_layout()
plt.show()
