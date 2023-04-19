import matplotlib.pyplot as plt
import numpy as np

# PI = np.pi
# t = np.linspace(-4*PI, 4*PI, 300)
# sin = np.sin(t)
#
# fig, ax = plt.subplots(figsize=(10,10))
#
# for ax_idx in range(12):
#     label_template = 'added by {}'
#     ax.plot(t, sin+ax_idx,
#             label=label_template.format(ax_idx)) # 이렇게도 포매팅할 수 있군..
#
# ax.legend(fontsize=15)
#
# plt.show()

# ---------------------------------------------------------------------------

# # legend의 열개수도 설정 가능
#
# PI = np.pi
# t = np.linspace(-4*PI, 4*PI, 300)
# sin = np.sin(t)
#
# fig, ax = plt.subplots(figsize=(10, 10))
#
# for ax_idx in range(12):
#     label_template = 'added by {}'
#     ax.plot(t, sin+ax_idx,
#             label=label_template.format(ax_idx))
# ax.legend(fontsize=15,
#           ncol=2)
#
# plt.show()

# -----------------------------------------

# bbox_to_anchor : 그래프 기준으로 위치 설정이 되는게 아니라 legend기준으로 위치가 바뀜.
# location을 center left로 하고(legend), bbox_to_anchor 위치로 legend의 center left가 감.
# bbox_to_anchor > 0~1 사이 값

# n_data = 100
# random_noise1 = np.random.normal(0, 1, (n_data, ))
# random_noise2 = np.random.normal(1, 1, (n_data, ))
# random_noise3 = np.random.normal(2, 1, (n_data, ))
#
# fig, ax = plt.subplots(figsize=(10, 7))
# ax.tick_params(labelsize=20)
#
# ax.plot(random_noise1,
#         label='random noise1')
# ax.plot(random_noise2,
#         label='random noise2')
# ax.plot(random_noise3,
#         label='random noise3')
#
# ax.legend(fontsize=20,
#           bbox_to_anchor=(1, 0.5),
#           loc='center left')
# fig.tight_layout()
#
# plt.show()


# -----------------------------------------------------------------

# PI = np.pi
# t = np.linspace(-4*PI, 4*PI, 300)
# sin = np.sin(t)
#
# fig, ax = plt.subplots(figsize=(10,10))
#
# for ax_idx in range(12):
#     label_template = 'added by {}'
#     ax.plot(t, sin+ax_idx,
#             label=label_template.format(ax_idx))
# ax.legend(fontsize=15,
#           ncol=4,
#           bbox_to_anchor=(0.5, -0.05),   # 마이너스 값 넣으면..
#           loc='upper center')
# fig.tight_layout()
#
# plt.show()

# ------------------------------------------------------------------

# # line style, marker 설정해주면 legend에도 적용이 됨
#
# PI = np.pi
# t = np.linspace(-4*PI, 4*PI, 50)
# sin = np.sin(t)
#
# fig, ax = plt.subplots(figsize=(10,10))
#
# ax.plot(t, sin,
#         label='sin(t)')
# ax.plot(t, sin+1,
#         marker='o',
#         label='sin(t)+1',
#         linestyle=':')
# ax.plot(t, sin+2,
#         marker='D',
#         label='sin(t)+2',
#         linestyle='--')
# ax.plot(t, sin+3,
#         marker='s',
#         label='sin(t)+3',
#         linestyle='-.')
# ax.legend(loc='center left',
#           bbox_to_anchor=(1,0.5),
#           fontsize=20)
# fig.tight_layout()
#
# plt.show()

# ---------------------------------------------------------------------
import matplotlib.cm as cm

# np.random.seed(0)
#
# n_class = 5
# n_data = 30
# center_pt = np.random.uniform(-20, 20, (n_class, 2)) # center point
# # print(center_pt)
# cmap = cm.get_cmap('tab20')
# colors = [cmap(i) for i in range(n_class)] # 5개 색깔
#
# # dict comprehension
# data_dict = {'class'+str(i): None for i in range(n_class)} # key:value
# # print(data_dict) # {'class0': None, 'class1': None, 'class2': None, 'class3': None, 'class4': None}
# for class_idx in range(n_class):
#     center = center_pt[class_idx]
#
#     x_data = center[0] + 2*np.random.normal(0, 1, (1, n_data)) # 이 식을 왜 이렇게 세웠을까? > center[0] 값을 중심으로 noise들을 만들어줌! > for문 한번돌 때, 하나의 noise cluster들이 생성됨!
#     y_data = center[1] + 2*np.random.normal(0, 1, (1, n_data))
#     data = np.vstack((x_data, y_data))
#
#     data_dict['class' + str(class_idx)] = data
# # print(center_pt[0][0]) # center = center_pt[class_idx] > center[0]
# # print(center_pt[0][1])
# # print(data_dict)
#
# fig, ax = plt.subplots(figsize=(12, 10))
# for class_idx in range(n_class):
#     data = data_dict['class' + str(class_idx)]
#     ax.scatter(data[0], data[1],
#                s=1000,
#                facecolor='None',
#                edgecolor=colors[class_idx],
#                linewidth=5,
#                alpha=0.5,
#                label='class'+str(class_idx))
# ax.legend(loc='center left',
#           bbox_to_anchor=(1, 0.5),
#           fontsize=10,
#           ncol=2)
# fig.tight_layout()
#
# plt.show()

# --------------------------------------------------------------

np.random.seed(0)

n_class = 5
n_data = 30
center_pt = np.random.uniform(-20, 20, (n_class,2))
cmap = cm.get_cmap('tab20')
colors = [cmap(i) for i in range(n_class)]

data_dict = {'class'+str(i) : None for i in range(n_class)}

for class_idx in range(n_class):
    center = center_pt[class_idx]

    x_data = center[0] + 2*np.random.normal(0, 1, (1, n_data))
    y_data = center[1] + 2*np.random.normal(0, 1, (1, n_data))
    data = np.vstack((x_data,y_data))

    data_dict['class'+str(class_idx)] = data

fig, ax = plt.subplots(figsize=(12,10))
for class_idx in range(n_class):
    data = data_dict['class'+str(class_idx)]
    ax.scatter(data[0], data[1],
               s=1000,
               facecolor='None',
               edgecolor=colors[class_idx],
               linewidth=5,
               alpha=0.5,
               label='class'+str(class_idx))
ax.legend(loc='upper right',
          bbox_to_anchor=(1,1),
          fontsize=30,
          ncol=2,
          title='Classes',
          title_fontsize=20,
          edgecolor='None',
          facecolor='None',
          labelspacing=1,                 # 위아래
          columnspacing=5)
fig.tight_layout()
plt.show()


































