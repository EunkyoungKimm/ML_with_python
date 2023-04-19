# KNN 구현해보자!
# 최종 목적은 decision boundary 나오게끔! k값 바꿔서 boundary 어떻게 변하는지 확인하기!
# list in list하셔서 pandas dataframe으로 하심

import matplotlib.pyplot as plt
import numpy as np

# data setting
athletes = [[2.50, 6.00, 'no'],
            [3.75, 8.00, 'no'],
            [2.25, 5.50, 'no'],
            [3.25, 8.25, 'no'],
            [2.75, 7.50, 'no'],
            [4.50, 5.00, 'no'],
            [3.50, 5.25, 'no'],
            [3.00, 3.25, 'no'],
            [4.00, 4.00, 'no'],
            [4.25, 3.75, 'no'],
            [2.00, 2.00, 'no'],
            [5.00, 2.50, 'no'],
            [8.25, 8.50, 'no'],
            [5.75, 8.75, 'yes'],
            [4.75, 6.25, 'yes'],
            [5.50, 6.75, 'yes'],
            [5.25, 9.50, 'yes'],
            [7.00, 4.25, 'yes'],
            [7.50, 8.00, 'yes'],
            [7.25, 5.75, 'yes']]
# test data (Speed :6.75, Agility:3.00)
test_data = [6.75, 3.00]

fig, ax = plt.subplots(figsize=(7,7))

ax.set_xlabel('Speed', fontsize=15)
ax.set_ylabel('AGILITY', fontsize=15)

for athlete in athletes:
    if athlete[2] == 'no':
        ax.scatter(athlete[0], athlete[1],
                   s=50, marker='^', color='red')
    else:
        ax.scatter(athlete[0], athlete[1],
                   s=50, marker='+', color='blue')

ax.scatter([], [],
           s=50, marker='+', color='blue',
           label='yes')
ax.scatter([], [],
           s= 50, marker='^', color='red',
           label='no')

ax.legend()

fig.tight_layout()

plt.show()

# ------------------------------------------------------------------------------------

# 나중에 numpy로 해보기

# # data setting
# athletes = [[2.50, 6.00, 'no'],
#             [3.75, 8.00, 'no'],
#             [2.25, 5.50, 'no'],
#             [3.25, 8.25, 'no'],
#             [2.75, 7.50, 'no'],
#             [4.50, 5.00, 'no'],
#             [3.50, 5.25, 'no'],
#             [3.00, 3.25, 'no'],
#             [4.00, 4.00, 'no'],
#             [4.25, 3.75, 'no'],
#             [2.00, 2.00, 'no'],
#             [5.00, 2.50, 'no'],
#             [8.25, 8.50, 'no'],
#             [5.75, 8.75, 'yes'],
#             [4.75, 6.25, 'yes'],
#             [5.50, 6.75, 'yes'],
#             [5.25, 9.50, 'yes'],
#             [7.00, 4.25, 'yes'],
#             [7.50, 8.00, 'yes'],
#             [7.25, 5.75, 'yes']]
# datas = np.array(athletes)
# # print(datas)
# # print(type(datas))
#
# fig, ax = plt.subplots(figsize=(7,7))
#
# ax.set_xlabel('Speed', fontsize=15)
# ax.set_ylabel('AGILITY', fontsize=15)
#
# for athlete in athletes:
#     if athlete[2] == 'no':
#         ax.scatter(athlete[0], athlete[1],
#                    s=50, marker='^', color='red')
#     else:
#         ax.scatter(athlete[0], athlete[1],
#                    s=50, marker='+', color='blue')
#
# ax.scatter([], [],
#            s=50, marker='+', color='blue',
#            label='yes')
# ax.scatter([], [],
#            s= 50, marker='^', color='red',
#            label='no')
#
# ax.legend()
#
# fig.tight_layout()
#
# plt.show()