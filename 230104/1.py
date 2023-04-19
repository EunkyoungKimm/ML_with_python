# colors - matplotlib은 tab10 색을 디폴트로 사용(라이브러리마다 디폴트 컬러 다름. 나중에는 사용된 색만 봐도 어떤 라이브러리인지 알 수 있음)
# 1) color list에 색깔을 넣어주고 사용
# for문에서 인덱스와 값을 같이 가져오는 함수 = enumerate

import matplotlib.pyplot as plt

# # 1
# color_list = ['b', 'g', 'r', 'c', 'm', 'y']
#
# fig, ax = plt.subplots(figsize=(5, 10))
# ax.set_xlim([-1, 1])
# ax.set_ylim([-1, len(color_list)])
#
# for c_idx, c in enumerate(color_list):
#     ax.text(0, c_idx,
#             "color="+c,
#             fontsize=20,
#             ha='center',
#             color=c)
# plt.show()

# ----------------------------------------------------------

# # 2
#
# color_list = ['tab:blue', 'tab:orange',
#               'tab:green', 'tab:red',
#               'tab:purple', 'tab:brown',
#               'tab:pink', 'tab:gray',
#               'tab:olive', 'tab:cyan']
#
# fig,ax = plt.subplots(figsize=(5,10))
# ax.set_xlim([-1, 1])
# ax.set_ylim([-1, len(color_list)])
#
# for c_idx, c in enumerate(color_list):
#     ax.text(0, c_idx,
#             "color="+c,
#             fontsize=20,
#             ha='center',
#             color=c)
# plt.show()

# ------------------------------------------------------------

# # 3
# # rgb값이 각각 0~255사이의 값을 가짐! -> 256가지
# # matplotlib에서는 0~1사이를 가짐!
#
# color_list = [(1., 0., 0.),       # 원래 제일 진한 R빨강 (255,0,0) > 빨간값만 제일 많이 가지고 있으니까
#               (0., 1., 0.),       # G
#               (0., 0., 1.)]       # B
#
# fig,ax = plt.subplots(figsize=(5,10))
# ax.set_xlim([-1, 1])
# ax.set_ylim([-1, len(color_list)])
#
# for c_idx, c in enumerate(color_list):
#     ax.text(0, c_idx,
#             f"color={c}",
#             fontsize=20,
#             ha='center',
#             color=c)
# plt.show()

# --------------------------------------------------------------

import matplotlib.pyplot as plt
import matplotlib.cm as cm

# # 4
# # cm > matplotlib의 colormap을 사용하겠다!
# import matplotlib.pyplot as plt
# import matplotlib.cm as cm
#
# cmap = cm.get_cmap('tab10', lut=20) # colormap에서 tab10의 20개를 뽑아오겠다 > cmap이라는 변수에 20개의 색이 저장됨 (lut = look up table) /  https://helloimmelie.tistory.com/3 (참고)
# fig, ax = plt.subplots(figsize=(8,8))
# for i in range(12):
#     ax.scatter(i, i, color=cmap(i), s=100) # 앞에는 data position(좌표!) >> (0,0),(1,1).... / cmap(i)가 for문 활용해서 인덱싱하는 것 같지만 함수! (f(0), f(1)과 형태 비슷) / s는 점의 크기
# plt.show()

# -------------------------------------------------------------

# # 5
# # 이제 continuous한 color를 써보자! (지금까지는 descriptive한 color를 씀)
# # (R, G, B, A) > A는 투명도
#
# n_color = 10
# cmap = cm.get_cmap('rainbow', lut=n_color) # n_color만큼 쪼개기
#
# for c_idx in range(n_color):
#     print(cmap(c_idx))

# --------------------------------------------------------------

# # 6
#
# n_color = 10 # 30, 100으로 바꿔보기!
# cmap = cm.get_cmap('rainbow', lut=n_color)
#
# fig, ax = plt.subplots(figsize=(15,10))
# ax.set_xlim([-1,1])
# ax.set_ylim([-1, n_color])
#
# for c_idx in range(n_color):
#     color = cmap(c_idx)
#
#     ax.text(0, c_idx,
#             f"color={cmap(c_idx)}", # "color="+cmap(c_idx)
#             fontsize=15,
#             ha='center',
#             color=color)
# plt.show()

# -------------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np

# # 7
# # data set generation (random한 데이터셋을 만든거임)
# # 정규분포로 random데이터 뽑아내기 (평균, 표준편차, ) > 평균이0이고 표준편차가 1인 정규분포는 '표준정규분포'이다
# # np.random.seed(0) >> 랜덤 시드 0에 해당되는 데이터들이 뽑힘. 그래서 몇번 돌려도 그래프 모양 안 바뀜(원래는 이거 없으면 random하게 데이터가 뽑히니까..바뀌는데)몇 번 코드를 돌려도 그래프 똑같이 나옴(괄호 안에 0혹은 42을 많이씀. 사실 아무거나 상관x)
# # np.random.seed 없으면 scatter 그래프 모양 계속 바뀜 ( 그래서 시드를 고정해줌)
# # 표준정규분포를 따르는 데이터 100개, 이걸 가지고 scatterplot 그림 /
# # plot은 주로 line plot을 그릴때 / marker모양을 plot에 넣어주면 scatter plot처럼 그려짐.
# # 마커 종류: o, v, ^, < , >, 8, s, p, *, h, H, D, d, P, X 등등..
# np.random.seed(0)
#
# n_data = 100
# x_data = np.random.normal(0, 1, (n_data,)) # 콤마..벡터라는거 나타내려고..습관처럼 쓰심
# y_data = np.random.normal(0, 1, (n_data,))
#
# fig, ax = plt.subplots(figsize=(7,7))
# ax.scatter(x_data, y_data) # scatter > scatter(x_data, y_data) = plot(x_data, y_data, 'o')
# # ax.plot(x_data, y_data) # plot - marker 설정x > 다 선으로 연결..난잡..
# # ax.plot(x_data, y_data,'o')
# # ax.plot(x_data, y_data, 'v')
#
# plt.show()

# --------------------------------------------------------------------

# # 8
# # 마커 사이즈
#
# np.random.seed(0)
#
# n_data = 100
# x_data = np.random.normal(0, 1, (n_data,))
# y_data = np.random.normal(0, 1, (n_data,))
#
# fig, ax = plt.subplots(figsize=(7,7))
# # scatter
# ax.scatter(x_data, y_data,
#            s=100)
#
# # # plot
# # ax.plot(x_data, y_data,
# #         'o', markersize=10)
#
#
# plt.show()

# ------------------------------------------------------------------

# # 색깔
#
# np.random.seed(0)
#
# n_data = 100
# x_data = np.random.normal(0, 1, (n_data,))
# y_data = np.random.normal(0, 1, (n_data,))
#
# fig, ax = plt.subplots(figsize=(7,7))
# # ax.scatter(x_data, y_data,
# #            s=100,
# #            color='r')
#
# ax.plot(x_data, y_data,
#         'o',
#         color='r',
#         markersize=10)
#
# plt.show()

# ----------------------------------------------------------------
# 최대, 최소 안에서 n개 데이터 만큼 뽑아주는데.. uniform은 모든 실수가 나올 확률이 동일한것..
# linspace > 똑같은 간격으로 뽑힘..(2니까 -5,5가 뽑힘)
# linear regression이라고 생각하면 됨(선형회귀)
# 보통은 공부시간, 수학성적이 있다고 했을 때 보통은 공부시간 비례해서 수학성적이 나옴..(x공부, y수학)
# scatter가 그 데이터 다 찍어본거고, 빨간선은 이런 경향성을 띄구나.. 그래서 예측가능해짐(모델을 학습시키고 나서 그 예측값을 line plot으로 그려준거임)
# data generation: 실제 데이터가 있으면 그 데이터를 넣어주면 되는 자리 / 모델을 실제 데이터로 학습시키기 전에 이렇게 data를 만들어서 모델의 성능을 판단할 때 사용하려고 data generation을 많이 함.


# np.random.seed(0)
#
# x_min, x_max = -5, 5
# n_data = 300
#
# # data generation
# x_data = np.random.uniform(x_min, x_max, n_data)
# y_data = x_data + 0.5*np.random.normal(0, 1, n_data) # 뒷부분 없으면 scatter plot이 직선 그대로.. 즉, 뒷부분을 통해 noise를 준거임. / noise를 생성해줄때는 정규분포를 이용해 만들어줌.
#
# pred_x = np.linspace(x_min, x_max, 2) # pred_x > 2개
# # print(pred_x)
# pred_y = pred_x # pred_y > 2개
#
# fig, ax = plt.subplots(figsize=(10,10))
# ax.scatter(x_data, y_data) # 색깔 따로 지정x > 디폴트값인 tab10의 첫번째 색으로 됨
#
# ax.plot(pred_x, pred_y,          # 2개의 점을 잇는 선그래프
#         color='r',
#         linewidth=3)
# plt.show()

# ----------------------------------------------------------------

# n_data = 10
# x_data = np.linspace(0, 10, n_data)
# y_data = np.linspace(0, 10, n_data)
#
# s_arr = np.linspace(10, 500, n_data) # size array
#
# fig, ax = plt.subplots(figsize=(10,10))
# ax.scatter(x_data, y_data,
#            s=s_arr)
#
# plt.show()

# -----------------------------------------------------------------

# # r,g,b값이 다 같으면 아래와 같은 색이 나옴! (무채색계열)(gray scale)
#
# n_data = 10
# x_data = np.linspace(0, 10, n_data)
# y_data = np.linspace(0, 10, n_data)
#
# # c_arr = [(c/10, c/10, c/10) for c in range(n_data)] >> c/10대신 c/n_data 사용하기!
# c_arr = [(c/n_data, c/n_data, c/n_data) for c in range(n_data)]
#
# fig, ax = plt.subplots(figsize=(10,10))
# ax.scatter(x_data, y_data,
#            s=500,
#            c=c_arr)
# plt.show()

# ------------------------------------------------------------------

# n_data = 10
# x_data = np.linspace(0, 10, n_data)
# y_data = np.linspace(0, 10, n_data)
#
# s_arr = np.linspace(10, 500, n_data)
# c_arr = [(c/n_data, c/n_data, c/n_data) for c in range(n_data)]
#
# fig, ax = plt.subplots(figsize=(10, 10))
# ax.scatter(x_data, y_data,
#            s=s_arr,
#            c=c_arr)
# plt.show()

# ----------------------------------------------------------------

# np.random.seed(0)
#
# n_data = 500
# x_data = np.random.normal(0, 1, size=(n_data,)) # size = 몇개
# y_data = np.random.normal(0, 1, size=(n_data,))
# s_arr = np.random.uniform(100, 500, n_data)
# c_arr = [np.random.uniform(0, 1, 3)
#          for _ in range(n_data)]
# # print(c_arr[:3])
#
# fig, ax = plt.subplots(figsize=(10,10))
# ax.scatter(x_data, y_data,
#            s=s_arr,
#            c=c_arr,
#            alpha=0.3)
# plt.show()

# ----------------------------------------------------------------

# PI =np.pi # np.pi = 3.141592.. > 계속 쓰기 힘드니까 변수에 할당에서 많이 씀
# n_point = 100
# t = np.linspace(-4*PI, 4*PI, n_point)
# sin = np.sin(t) # sin그래프 그리기 위해 sin값
#
# cmap = cm.get_cmap('Reds', lut=n_point) # colormap을 가져올때 (우리가 필요한 만큼)n_point만큼 쪼개서 가져오는 것!
# c_arr = [cmap(c_idx) for c_idx in range(n_point)] # 아래서 사용하려면 이렇게 인덱싱. 지정해줘야 함.
#
# fig, ax = plt.subplots(figsize=(15,10))
# ax.scatter(t, sin,
#            s=300,
#            c=c_arr)
# plt.show()

# -----------------------------------------------------------------
# # advanced markers: facecolor / edgecolor / linewidth 등 설정 가능
#
# fig, ax = plt.subplots(figsize=(5,5))
#
# # ax.scatter(0, 0,
# #            s=10000)
#
# # ax.scatter(0, 0,
# #            s=10000,
# #            facecolor='r')
#
# # ax.scatter(0, 0,
# #            s=10000,
# #            facecolor='r',
# #            edgecolor='b')
#
# ax.scatter(0, 0,
#            s=10000,
#            facecolor='r',
#            edgecolor='b',
#            linewidths=5)
#
# plt.show()

# ----------------------------------------------------------------

# # facecolor='white'
# n_data = 100
# x_data = np.random.normal(0, 1, (n_data,))
# y_data = np.random.normal(0, 1, (n_data,))
#
# fig, ax = plt.subplots(figsize=(5,5))
#
# ax.scatter(x_data, y_data,
#            s=300,
#            facecolor='white',
#            edgecolor='tab:blue',
#            linewidths=5)
#
# plt.show()
#
# # facecolor='None' > 겹쳐진거까지 다 보임
# n_data = 100
# x_data = np.random.normal(0, 1, (n_data,))
# y_data = np.random.normal(0, 1, (n_data,))
#
# fig, ax = plt.subplots(figsize=(5,5))
#
# ax.scatter(x_data, y_data,
#            s=300,
#            facecolor='None',
#            edgecolor='tab:blue',
#            linewidths=5)
#
# plt.show()

# -----------------------------------------------------------------

np.random.seed(0)

n_data = 200
x_data = np.random.normal(0, 1, (n_data,))
y_data = np.random.normal(0, 1, (n_data,))

fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(x_data, y_data,
           s=300,
           facecolor='None',
           edgecolor='tab:blue',
           linewidths=5,
           alpha=0.5)
plt.show()



















