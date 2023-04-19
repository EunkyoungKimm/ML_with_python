# 1-Dimensional Window Extraction
'''
- Window란?
  10개의 원소를 가진 1차원 벡터가 주어졌을때, 3칸의 window가 차례대로 지나가면서(이동하면서) 데이터를 3개씩 뽑는 과정
  **공식 외우기!!! => window의 개수(L') = L - W + 1 (L은 데이터의 개수!)(L'=L프라임)

- 이제 window를 구현해보자!
'''

# # 1차원 원소(벡터에서)에서 window를 뽑아내보자!
#
# import numpy as np
#
# data = 10*np.arange(1, 11)
# L = len(data)
# W = 3
# print(data, '\n')
#
# L_ = L - W + 1    # 8
# for idx in range(L_):
#     print(data[idx:idx + W]) # [0 : 0 + 3] 부터 [7 : 7 + 3] 까지 slicing


# ---------------------------------------------------------------------------

# # 2-Dimensional Window Extraction
# # 행렬 형태 데이터에서 window 뽑아내는 것!(window도 3x3짜리 데이터!)
# # width에 들어갈 window수랑, height 들어갈 window 수랑 구해서 곱하면 됨!
# # > data.shape을 먼저 뽑아와야 해!!
#
# import numpy as np
#
# data1 = 10*np.arange(1,8).reshape(1,-1)
# data2 = 10*np.arange(5).reshape(-1,1)
# print(data1)
# print(data2)
# data = data1 + data2 # 이게 신기!! data1 => 10~70 / data2 => 0~40 설정 잘 해주기!
# print(data, '\n')
#
# H, W = data.shape
# F = 3 # 임시적으로 window size를 의미
# H_ = H - F + 1
# W_ = W - F + 1
#
# for h_idx in range(H_):
#     for w_idx in range(W_):
#         print(data[h_idx : h_idx + F, w_idx : w_idx + F])
# --------------------------------------------------------------------------------------------------------------------
# 1-D Correlation
# Correlation 이게 진짜 중요해!!!!!!!!!!!!!!
# sobel filtering 자체는 딥러닝은 아니고 image 처리 기법임. 근데 이것의 연산 원리가 correlation임!! (이게 cnn동작원리!, 인공뉴러 동작원리! 그래서 딥러닝 동작 원리 알 수 있음!)
'''
- correlation?
  window를 쭉 뽑으면서, 내가 원하는 패턴이 있는 부분을 추출하는 연산!
  1. data에서 window(데이터에서 3x3 뽑힌거.. 파란색 색깔 칠해진거 다 window임)를 뽑고, filter와 '내적(dot product)'한 결과를 저장 (내적 : 원소끼리 곱해서 더한 값)
  2. window와 filter가 같을 때 가장 큰 값을 출력 ex. filter가 [-1, 1, -1] 이면 window가 [-1, 1, -1]일 때 가장 큰 값!
  3. window와 filter가 반대 벡터일 때 가장 작은 값 출력 ex. filter가 [-1, 1, -1] 이면 window가 [1, -1, 1]일 때 가장 작은 값!
  
- 딥러닝의 인공뉴런이 하는 일이 비슷! 어떤 임계점 이상으로 비슷한 애들을 추출하는게 얘의 역할임! correlation이랑 ==
'''

# # 1-D Correlation
#
# import numpy as np
#
# np.random.seed(0)
#
# data = np.random.randint(-1, 2, (10, ))
# filter_ = np.array([-1, 1, -1])                  # 파이썬 내장 함수 중 filter가 있어서, 충돌 막기 위해서 꼭 underscore(_) 써주기!
# print(f'{data = }')
# print(f'{filter_ = }')
#
# L = len(data)
# F = len(filter_)
#
# L_ = L - F + 1
# filtered = []
# for idx in range(L_):
#     window = data[idx : idx + F]
#     filtered.append(np.dot(window, filter_)) # np.dot => 내적
#     # NOTICE
#     # window는 for문을 지나며 항상 바뀌는 부분
#     # filter_는 항상 [-1,1,-1]로 일정
#
# filtered = np.array(filtered)
# print('filtering result:', filtered)


# --------------------------------------------------------------------------------------------------------------------

# 2-D Correlation
# - 왼쪽 위부터 아래쪽 오른쪽까지 모든 window마다 filter랑 원소별 곱셈을 하고, 다 더하기
# - 이 결과를 저장하면 (3,5) 행렬이 만들어짐

import numpy as np

data1 = 10*np.arange(1,8).reshape(1, -1)
data2 = 10*np.arange(5).reshape(-1,1)

data = data1 + data2

filter_ = np.array([[1, 2, 5],
                   [-10, 2, -2],
                   [5, 1, -4]])
print(data, '\n')

H, W = data.shape
F = filter_.shape[0] # 임시적으로 window size를 의미
H_ = H - F + 1
W_ = W - F + 1

filtered_data = np.zeros(shape=(H_, W_))
for h_idx in range(H_):
    for w_idx in range(W_):
        window = data[h_idx : h_idx + F, w_idx : w_idx + F]
        z = np.sum(window * filter_)

        filtered_data[h_idx, w_idx] = z
print(filtered_data)

# ------------------------------------------------------------------------------------------------------------------

# 수업시간에 구현한 1D Correlation을 for문 없이 만들어보세요.
# => 연산 속도!
# (이미지처리나 딥러닝에서는 for문 하나 없애는 거에 따라서 천차만별임! for문 없이 numpy로 하면 연산속도가 700배까지 나는걸 봤다고 함..)
# 사진))))   SSD(저장공간..) / RAM(한번에 돌릴수 있는 프로그램..수..?) / CPU(실행시킴..) / CACHE
# ram에서 가져온 데이터가 cache.. 우리가 원하는게 딱 모아져 있으면 cache>cpu 이 연산 시간 밖에 안걸림!
# 근데 리스트는 데이터를 모아서 저장하는게 아니라 ram에서 빈공간을 찾아서 넣어줌.(산발적으로 저장) > for문 써서 리스트 처리하면 굉장히 오래걸림!(그 데이터 찾는데만 해도 오래걸림)
# 근데 numpy는 array를 만들어줌. 그 array를 하나의 덩어리로 저장해줌! > numpy는 메모리 layout까지 설계를 해준 라이브러리임!!
# 그래서 numpy ndarray를 만들어서 실행시켜주면 연산 속도가 빠른거임!
# 리스트로 저장하면 ram과 cache 사이를 왔다갔다 하는게 엄청 많음!
# numpy로 리스트 접근은? (for문 사용하지 않고) > numpy만 쓰는거랑, numpy+list쓰는거랑 또 속도 차이가 많이남.
# 파이썬 코드가 그대로 cpu에 들어가는게 아니라 머신 코드(어셈블리 언어?)로 변환하는 과정이 필요함. for문을 iterate할때마다 그 변환 명령어가 엄청 많음, numpy한줄 바로 이므로.. 훨씬 짧음!
# 연산속도 줄이려면 > 코드에서 for문을 최대한 줄이는 것! numpy를 사용할 것!!

# np.matmul => 행렬의 곱
# 결과 : [ 2 -1  1 -1  2 -3  3 -1]
# np.matmul => size가 같아야 함 => data를 3개씩 나눠주려면..?

# import numpy as np
#
# np.random.seed(0)
#
# data1 = np.random.randint(-1, 2, (10, ))
# data2 = np.zeros(8).reshape(-1,1)
#
# data = data1 + data2
# filter_ = np.array([-1, 1, -1])
#
#
#
# print(data)
#
# filtered_data = np.matmul(data, filter_)
# print(filtered_data)

# ------------------------------------------------
# import numpy as np
#
# data1 = 10*np.arange(1,8).reshape(1,-1)
# filter_ = np.array([-1, 1, -1])
#
# L = len(data1[0])
# F = len(filter_)
# L_ = L - F + 1
#
# data2 = 10*np.arange(L_).reshape(-1,1)
#
# data = data1 + data2 # 브로드캐스팅 됨..흠 근데 10의 배수니까 가능한거 아님..?
#
# data = data[:, 0:3]
#
# filtered_data = np.matmul(data, filter_)
# print(filtered_data)

# ------------------------------------------------------------------------------------------

# 10의 배수 아니여도 가능하게 끔!

# import numpy as np
#
# np.random.seed(0)
#
# data = np.random.randint(-1, 2, (10,))
# filter_ = np.array([-1, 1, -1])
#
# L, F = len(data), len(filter_)
# L_ = L - F + 1
#
# filter_idx = np.arange(F).reshape(1, -1) # 얘를 만들어줘야 함!
# window_idx = np.arange(L_).reshape(-1, 1)
# idx_arr = filter_idx + window_idx
# # print(filter_idx,'\n',window_idx, '\n', idx_arr, '\n')
# window_mat = data[idx_arr] # 이렇게 하면 브로드 캐스팅 됨!!
# # print(window_mat)
# print(window_mat.shape, filter_.shape) # (8, 3), (3,)
#
# # 행렬의 곱셈을 이용하기 위해서 filter_를 (3, 1)로 바꿔줄 필요가 있음
# correlations = np.matmul(window_mat, filter_.reshape(-1, 1))
#
# # 만약 결과를 vector처럼 다뤄야 하면
# correlations = correlations.flatten()
# print(correlations)


# 질문
# Q. filter_.shape에서 이미 (3,)이렇게 나왔는데 reshape해서 (3,1)로 바꿔주는 이유가 행렬의 곱셈 사용해야 해서 벡터를 행렬로 바꾸려고 reshape한건가요???
# A. 네, reshape을 안해줘도 결과는 잘 나오기는 하지만 차원이 확장되거나 하는 경우에는 꼭 data 나 filter의 shape을 프린트해보고
# 맞춰주어야 할 때도 있어요~! 여기서는 어차피 지금 1차원 벡터라서 자동적으로 브로드캐스팅에 의해서 연산이 되었지만 맞춰주어야 하는 경우도 있습니다~!








