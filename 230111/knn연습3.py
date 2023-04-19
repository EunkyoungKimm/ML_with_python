# np.meshgrid 함수 이용해서 decision boundary 표현해보기
# (각 축마다 100개의 점을 뽑아서 총 10000개의 점을 생성해주는 거에요 .
# 그래서 각각 10000개의 점이 테스트 데이터가 되어서 KNN classification을 하고, 분류 결과에 따라 빨강/파랑으로 표현되도록!)


import matplotlib.pyplot as plt
import numpy as np

# data setting
# dataframe을 만들지 않고 numpy로 만들면.. 일자로 나오는데.. dataframe을 먼저 안만들어줘서 그런건가?
# data = [speed, agility, draft]
draft = ['yes', 'no']

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
test_data = [[6.75, 3.00]]

# euclidean distance
diff_speed = 0
diff_agility = 0
e_distance_dict = {}              # 리스트로 하니까 나중에 뭐가 뭔지 모름.. > dict
for idx in range(len(athletes)):
    diff_speed = (test_data[0][0] - athletes[idx][0]) ** 2
    diff_agility = (test_data[0][1] - athletes[idx][1]) ** 2
    e_distance = (diff_speed + diff_agility) ** 0.5

    e_distance_dict[e_distance] = idx
# print(e_distance_dict)

sort_distance = sorted(e_distance_dict.items())
# print(sort_distance)
# print(sort_distance[0][1])

# 거리가 가까운 k개의 점들과 test_data 점 잇기 위한 data setting
x = [test_data[0][0]]
y = [test_data[0][1]]
classes =[] # 테스트 데이터의 클래스 분류를 위해

k = 3
dots = sort_distance[:k]
for i in range(k):
    idx = sort_distance[i][1]
    x.append(athletes[idx][0])
    y.append(athletes[idx][1])
    classes.append(athletes[idx][2])


class1 = classes.count(draft[0])
class2 = classes.count(draft[1])
if class1 > class2:
    result = draft[0]
else:
    result = draft[1]


# decision boundary - 10000개의 점들이 테스트 데이터가 되어서! 경계선을 정해주는거임!#############################
a = np.linspace(1,9,100)
b = np.linspace(1,10,100)
db_x,db_y = np.meshgrid(a, b)

# plot
fig, ax = plt.subplots(figsize=(7,7))

ax.set_xlabel('Speed', fontsize=15)
ax.set_ylabel('AGILITY', fontsize=15)

plt.text(test_data[0][0]+0.1, test_data[0][1],
         s=result,
         color='green',
         fontsize=12)

# decision boundary
ax.scatter(db_x, db_y,
           alpha=0.1)

for athlete in athletes:
    if athlete[2] == 'no':
        ax.scatter(athlete[0], athlete[1],
                   s=50, marker='^', color='red')
    else:
        ax.scatter(athlete[0], athlete[1],
                   s=50, marker='+', color='blue')

ax.scatter(test_data[0][0], test_data[0][1],
           s=50, marker='*', color='green')

# 거리가 가까운 k개의 점
for i in range(k):
    line_x = [x[0], x[i+1]]
    line_y = [y[0], y[i+1]]
    ax.plot(line_x, line_y, color='pink')

# fake scatter for legend
ax.scatter([], [],
           s=50, marker='+', color='blue',
           label='yes')
ax.scatter([], [],
           s= 50, marker='^', color='red',
           label='no')


ax.legend()

plt.show()
