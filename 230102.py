import numpy as np

# # euclidean distance 구하기
#
# a = np.array((1,0,1))
# b = np.array((2,3,1))
#
# # method1
# square = np.square(a - b)
# print(square)
#
# sum_square = np.sum(square)
# print(sum_square)
#
# e_distance = np.sqrt(sum_square)
# print(e_distance)
#
# # method2 - 얘는 norm구하는거임
# distance = np.linalg.norm(a - b)
# print(distance)
#
# # ------------------------------------------
#
# # manhattan distance 구하기
#
# a = np.array((1,0,1))
# b = np.array((2,3,1))
#
# # method1
# abs_diff = np.abs(a - b)
#
# m_distance = np.sum(abs_diff)
# print(m_distance)
#
# # method2
#
# m_distance = np.sum(np.abs(a - b))
# m_distance

# ----------------------------------------------------------------------
# KNN (K-nearest-neighbor)
# k개의 가까운 이웃들(끼리끼리.. 이친구를 보면 너가 누군지 알 수 있다! 이런 알고리즘임)

# 운동선수들 데이터(draft 뽑히는지 안뽑히는지)
# 각각의 feature들이(speed, agility) 하나의 axis를 이룸
# 거리가 중요한 척도임! knn에서! (여기서 e_distance, m_distance사용)


# # euclidean distance
# d5, d12 = [2.75, 7.50], [5.00, 2.50]
#
# sum_square = 0
# for idx in range(len(d5)): # idx말고 dim이라고 변수명 정해도 좋을듯
#     sum_square += (d5[idx] - d12[idx]) ** 2
# e_distance = sum_square ** 0.5
# print(e_distance)
#
# # manhattan distance
# v1, v2 = [2.75, 7.50], [5.00, 2.50]
# abs_sum = 0
#
# for idx in range(len(v1)):
#     abs_sum += abs(d5[idx] - d12[idx])
# print(abs_sum)
#
#
#
# # euclidean distance with nympy
# a = np.array((2.75, 7.50))
# b = np.array((5.00, 2.50))
#
# square = np.square(a - b)
# sum_square = np.sum(square)
# e_distance = np.sqrt(sum_square)
#
# print(e_distance)
#
# # manhattan distance with numpy
# m_distance = np.sum(np.abs(a - b))
# m_distance
# print(m_distance)
#
# # list in list로 해보기!
#
# instances = [[2.75, 7.50],
#              [5.00, 2.50]]
#
# diff = 0
# for dim_idx in range(len(instances[0])):
#     diff += (instances[0][dim_idx] - instances[1][dim_idx]) ** 2
# e_distance = diff ** 0.5
# print('euclidean distance: ', e_distance)
#
# abs_sum = 0
# for dim_idx in range(len(instances[0])):
#     diff = instances[0][dim_idx] - instances[1][dim_idx]
#     if diff < 0:
#         abs_sum += -diff
#     elif diff >= 0:
#         abs_sum += diff
# m_distance = abs_sum
# print('manhattan distance: ', m_distance)

# ------------------------------------------------------------------------------
# # test instance와 나머지 벡터와의 거리
#
# instances = np.array([[5., 2.5, 3],
#                       [2.75, 7.50, 4],
#                       [9.10, 4.5, 5],
#                       [8.9, 2.3, 6]])
# test_instance = instances[0]
#
# distances = []
# for instance in instances:
#     distance = np.sqrt(np.sum((instance - test_instance)**2))
#     distances.append(distance)
# print(distances)
#
# # columnize로 봄(세로가 한 벡터)
#
# instances = np.array([[5., 2.5, 3],
#                       [2.75, 7.50, 4],
#                       [9.10, 4.5, 5],
#                       [8.9, 2.3, 6]])
# test_instance = instances[:, 0] # [행, 열] > 행은 다 가져오고 열 중에는 0번 index만 가져와라.
# print(instances.shape) # (4,3)
#
# n_cols = instances.shape[1] # 3
# distances = []
# for col_idx in range(n_cols):
#     instance = instances[:, col_idx]
#     distance = np.sqrt(np.sum((instance - test_instance)**2))
#     distances.append(distance)
# print(distances)

# --------------------------------------------------------------------
# minkowski distance > manhattan, euclidean 등을 p값에 따라 일반화 시켜 놓은 것 > 적절한 p값을 찾는게 중요!(knn에서는 보통 euclidean을 씀)

# d12, d17

instances = [[5.25, 9.50],
             [5.00, 2.50]]

# euclidean
diff = 0
for dim_idx in range(len(instances[0])):
    diff += (instances[0][dim_idx] - instances[1][dim_idx]) ** 2
e_distance = diff ** 0.5
print('euclidean distance: ', e_distance)

# manhattan
abs_sum = 0
for dim_idx in range(len(instances[0])):
    diff = instances[0][dim_idx] - instances[1][dim_idx]
    if diff < 0:
        abs_sum += -diff
    elif diff >= 0:
        abs_sum += diff
m_distance = abs_sum
print('manhattan distance: ', m_distance)

# manhattan은 같은데 euclidean은 다름! 왜 이런 차이가 날까? euclidean에는 제곱이 들어감! 하나의 feature 차이가 클수록 euclidean은 더 영향을 많이 받음! 즉, 민감하다.

# 새로운 데이터(speed=6.75, agility=3.00)
# 다른 데이터랑 거리 다 계산해주고, sorting한게 표! >> 18번이랑 비슷하다고 생각할 수 있음! >> target feature가 draft니까 새로운 데이터는 yes임!
# knn은 거리 계산해서 가까운거 보는거임!

# k값은 몇 개의 데이터를 볼거냐임. k는 항상 홀수!!!를 씀(짝수면.. 반반이면 판단x)
# voronoi tesellation : k가 1인 경우! (같은 공간에는 같은 target feature)
# decision boundary : target feature가 달라지는 경계

# knn장점: 새로운 데이터로 update가능.. decision boundary가 달라짐! update됨! 계속해서 새로운 데이터를 판단하고, 업데이트해서 모델을 계속 학습시킬 수 있음.
# 맨 위에 세모는 노이즈 데이터일 가능성이 큼!
# k = 3으로 높여주면 맨 위에 세모데이터가 있지만 새로운 데이터가 들어와도 경계가 잘 정해져있어서 옳게 분류가 될 수 있음!
# 그렇다고 너무 높여서 k=5해주면 제대로된 데이터도 학습을 잘 못함....
# 적절한 k 값을 찾는 것이 중요!!!!
# data target label개수가 서로 너무 다르면 학습이 잘 안 될 수 있음(13, 7)










