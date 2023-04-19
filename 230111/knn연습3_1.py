# np.meshgrid 함수 이용해서 decision boundary 표현해보기
# (각 축마다 100개의 점을 뽑아서 총 10000개의 점을 생성해주는 거에요 .
# 그래서 각각 10000개의 점이 테스트 데이터가 되어서 KNN classification을 하고, 분류 결과에 따라 빨강/파랑으로 표현되도록!)

# decision boundary - 10000개의 점들이 테스트 데이터가 되어서! 경계선을 정해주는거임!
# np.meshgrid 격자망 > 격자들이 생겨서 겹치는 부분에 그만큼 점들을 만들어 주는 것!


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

# decision boundary만들때 np써서.. np.array형태로 만들어주기!
athletes_np = np.array(athletes)
# print(athletes_np)
# feature_data = np.array([athletes_np[:,0], athletes_np[:,1]]) # 모든행의 0번째 열.. /대괄호..
speed_np = np.array(athletes_np[:, 0])
agility_np = np.array(athletes_np[:, 1])
# feature_np = np.array([speed_np, agility_np]) # 이렇게 하면 (2,20)이 나옴..
# print(np.shape(athletes_np), np.shape(feature_np))
# print(feature_np)
feature_np = np.hstack([speed_np.reshape(-1,1), agility_np.reshape(-1,1)])
# print(feature_np.shape)


# test data (Speed :6.75, Agility:3.00)
test_data = [[6.75, 3.00]]

# test data > euclidean distance
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


# decision boundary - 10000개의 점들이 ''테스트 데이터''가 되어서! 경계선을 정해주는거임!#############################
# numpy니까 for문 굳이x
a = np.linspace(1,9,100)
b = np.linspace(1,10,100)
# print(a.shape) # (100, ) 1은 생략..
# print(b.shape) # (100, ) 1은 생략..
db_x,db_y = np.meshgrid(a, b)
print(db_x.shape, db_y.shape) # (100, 100) (100, 100)
X = np.hstack([db_x.reshape(-1,1), db_y.reshape(-1,1)]) # 대괄호..

# print(np.shape(X))
#
# # type 맞추기
# print(feature_np.dtype) # <U32
# print(X.dtype) # float64

float_feature_np = feature_np.astype(np.float64)
# print(float_feature_np)
e_dists = []
for x_test in X:
    e_dist = np.sum((float_feature_np - x_test)**2, axis=1) # x_test자리에 X썼다가.. 와우 코드를 잘 읽자!!! / axis=1일때 원리 이해해보기..
    e_dists.append(e_dist)

e_dists_np = np.array(e_dists)

# print(e_dists_np)
# print(e_dists_np.shape) # (3,20) / 위에서 X[:3]일때(for x_test in X[:3])
close_e_dist = e_dists_np.argsort()
# print(close_e_dist)

# 가까운 점 k개
results = []
for i in range(len(close_e_dist)):
    r = []
    k_close = close_e_dist[i][:k]
    # print(k_close)
    for j in range(len(k_close)):
        idx = k_close[j]
        r.append(athletes[idx][2])
    results.append(r)

# print(results)

# draft찾기
s_lst = []
for i in range(len(results)):
    class1 = results[i].count(draft[0]) # yes
    class2 = results[i].count(draft[1]) # no
    if class1 > class2:
        r_s = 'blue'
        s_lst.append(r_s)
    else:
        r_s = 'red'
        s_lst.append(r_s)
# print(s_lst)






# plot
fig, ax = plt.subplots(figsize=(7,7))

ax.set_xlabel('Speed', fontsize=15)
ax.set_ylabel('AGILITY', fontsize=15)

plt.text(test_data[0][0]+0.1, test_data[0][1],
         s=result,
         color='green',
         fontsize=12)

# # # decision boundary
# for i in range(10000): # 이렇게 for문 돌릴 필요없!numpy똑똑하네...ㅎ / color에 s_lst도 알아서 돌려주면서 그림
#     ax.scatter(X[i,0], X[i,1],
#                color=s_lst[i],
#                alpha=0.2)
ax.scatter(X[:,0], X[:,1],
           color=s_lst,
           alpha=0.1)
print(X.shape)

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
    ax.plot(line_x, line_y, color='deeppink')

# fake scatter for legend
ax.scatter([], [],
           s=50, marker='+', color='blue',
           label='yes')
ax.scatter([], [],
           s= 50, marker='^', color='red',
           label='no')


ax.legend()

plt.show()

# 느낀점: numpy 사용법만 알았더라면,, dataframe 사용법만 알았더라면 코드가 훨씬 깔끔하고, 실행시간(for문)도 훨씬 빨랐을 것이다!!!! 공부하자!
# 데이터 다루는 방법 다양한 방법 잘 숙지하고 익히기!
# numpy하면 브로드캐스팅 되서 알아서 연산 쭉 해줘서 for문을 굳이 안써줘도 됨! 이걸 잘 이용하자!
