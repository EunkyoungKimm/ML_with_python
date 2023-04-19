import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

athletes = [[2.50, 6.00, 'no'],
            [3.75, 8.00, 'no'],
            [2.25, 5.50, 'no'],
            [3.25, 8.25, 'no'],
            [2.75, 7.25, 'no'],
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

athletes_df = pd.DataFrame(athletes,
                           columns=['SPEED', 'AGILITY', 'DRAFT'])
# print(athletes_df)

X = athletes_df[['SPEED', 'AGILITY']].to_numpy() # 여러 개의 열(multi-column) [[]]
y = athletes_df['DRAFT'].values # 한개의 열 []


# print(X)
# print(X.shape) # (20, 2)
# print(y, type(y)) # df.values >> <class 'numpy.ndarray'>
# b = athletes_df['DRAFT']
# z = athletes_df[['DRAFT']] # DRAFT라는 컬럼이 생성됨! / values안했으니까 df가 만들어짐. / values하면 값만 np.ndarray에 저장됨
# print(b, type(b)) # type = Series (시리즈 = 1개의 열) / 결과를 보면 딕셔너리의 key값이 인덱스(index)로, value값이 값으로 들어가 있음.(시리즈는 딕셔너리와 비슷!)
# print(z, type(z)) # type = DataFrame (데이터프레임 = 다수의 시리즈를 모은 것)
# cc = athletes_df[['SPEED', 'AGILITY']] # 컬럼이 SPEED와 AGILITY인 데이터프레임이 생성됨! /// 뒤에 to_numpy()나 values 하면 넘파이 어레이가 됨!!!!!
# print(cc)

fig, ax = plt.subplots(figsize=(10, 10))

X_pos, X_neg = X[y == 'yes'], X[y == 'no']
# print(type(X_pos)) # <class 'numpy.ndarray'>
# print(X_pos)

ax.scatter(X_pos[:, 0], X_pos[:, 1], color='blue', # speed, agility
           marker='+', s=130, label='yes')
ax.scatter(X_neg[:, 0], X_neg[:, 1], color='red',
           marker='^', s=100, label='no')
ax.set_xlabel("Speed", fontsize=20)
ax.set_ylabel("AGILITY", fontsize=20)
ax.legend(fontsize=15)
ax.tick_params(labelsize=15)

test_X = np.array([6.75, 3.00])
ax.scatter(test_X[0], test_X[1], color='fuchsia', marker='*', s=150)

K = 3

distances = np.sum((X - test_X)**2, axis=1) # **0.5 안함. 그냥 크기 비교니까.. 정확한 거리계산하는게 아니라 그냥 크기 비교이므로 다같이 안해주면 O
distances_argsort = np.argsort(distances)

# print(distances, distances.shape) # (20, ) / test_X = np.array([6.75, 3.00]) 랑 거리 구한거임
# print(distances_argsort)

close_K_indices = distances_argsort[:K] # index 복수형 > indices
# print(close_K_indices) # [17 11  9]
for close_x_idx in close_K_indices:
    close_x = X[close_x_idx]
    print("close_x:", close_x)
    ax.plot([close_x[0], test_X[0]],[close_x[1], test_X[1]], # ax.plot([a,b] ,[c,d]) >> 점(a,c)와 점(b,d)를 이은 line plot을 그려줌!
            color='fuchsia', lw=1, ls=':')

close_classes = y[close_K_indices]
# print(close_classes, type(close_classes), close_classes.dtype) # ['yes' 'no' 'no'] / <class 'numpy.ndarray'> / object >>> numpy니까 인덱스도 여러개 값 한번에 가능..wow...

unique, cnts = np.unique(close_classes, return_counts=True) # np.unique > 고유한 값들만 모아서 반환 / return_counts > 고유한 값들이 등장하는 횟수(True로 해주면 이것도 구해줌) // unique랑 cnts로 언패킹해준건가?브로드캐스팅인가?
# print(unique) # ['no' 'yes']
# print(cnts) # [2 1]
# # https://jimmy-ai.tistory.com/185 >>> np.unique 공부할 때 참고하기

classified_as = unique[np.argmax(cnts)]
# print(np.argmax(cnts)) # np.argmax > 최대값 위치(인덱스) 반환 / np.argmin > 최소값 위치(인덱스) 반환
# print(unique[0]) # no
# print(classified_as) # no

ax.text(test_X[0]+0.2, test_X[1], classified_as, color='fuchsia',
        fontsize=20)

x1_lim, x2_lim = ax.get_xlim(), ax.get_ylim() # ax.get_xlim() > x축 범위를 (a~b) 로 반환 >>> a를 최솟값, b를 최댓값으로 사용 가능!
x1 = np.linspace(x1_lim[0], x1_lim[1], 100)
x2 = np.linspace(x2_lim[0], x2_lim[1], 100)
# print(x1_lim) # (1.6875, 8.5625)
# print(x1)

X1, X2 = np.meshgrid(x1, x2)
test_X = np.hstack([X1.reshape(-1, 1), X2.reshape(-1, 1)])
# print(test_X.shape) # (10000, 2)
# print(len(test_X)) # 10000
# print(X, X.shape, len(X)) # [[speed agility]] / (20, 2) / 20
# print(X.shape, test_X.shape, test_X[1].shape)
preds = []
for test_x in test_X:
    distances = np.sum((X - test_x) ** 2, axis=1) # (20,) # test_x 하나랑 X 20개랑 연산해서 for문 한번에 20번 연산한 결과를 [~~(20개)~~] distances.shape = (20,) >> axis=1 행으로 계산/ axis=0하면 distances.shape = (2,) >> 열을 쭉계산(speed, agility 전체)
#     print(distances)
# print(distances.shape)
    distances_argsort = np.argsort(distances) # (20,)크기의 sorting된 값의 index를 보여줌
    close_K_indices = distances_argsort[:K] # [12 18 13]
    close_classes = y[close_K_indices]
    unique, cnts = np.unique(close_classes, return_counts=True)
    pred = unique[np.argmax(cnts)]

    if pred == 'yes': preds.append(1)
    elif pred == 'no': preds.append(0)

ax.scatter(X1, X2, c=preds, cmap='bwr_r', alpha=0.1) # cmap에 bwr_r하고, c에 숫자 넣어주면.. 그 숫자 개수 따라서 색 숫자가 알아서 정해져서 잘 분류해서 그려줌!!

plt.show()