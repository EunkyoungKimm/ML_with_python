# # 1-70) 홀수/짝수 구하기(2)
#
# numbers = list()
# for num in range(20):
#     numbers.append(num)
#
# numbers.append(3.14)
# print(numbers)
#
# for num in numbers:
#     if num % 2 == 0:
#         print('Even Number')
#     elif num % 2 == 1:
#         print('Odd Number')
#     else:
#         print('Not an Integer')

# # 1-71) n배수들의 합 구하기
#
# multiple_of = 3
#
# numbers = list()
# for num in range(100):
#     numbers.append(num)
#
# sum_multiple_of_n = 0
# for num in numbers:
#     if num % multiple_of == 0:
#         sum_multiple_of_n += num
# print(sum_multiple_of_n)

# # 1-72) 최댓값, 최솟값 구하기
#
# scores = [60, 40, 70, 20, 30]
# M, m = 0, 100 # 초기화 - 처음에 M,m이 60으로 됨! >> 이렇게 처음에 초기화를 하면 단점: error가 많이 남(리스트 안에 값들이 '점수'라는 것을 알기 때문에 우리가 이렇게 할 수 있었음. 하지만 음수라면, 100보다 크다면 error)
#
# for score in scores:
#     if score > M:
#         M = score
#     if score < m:
#         m = score
# print('Max value: ', M)
# print('min value: ', m)

# # 1-73)
#
# scores = [-20, 60, 40, 70, 120]
#
# # method1
# M, m = scores[0], scores[0]
# for score in scores:
#     if score > M:
#         M = score
#     if score < m:
#         m = score
# print('Max value: ', M)
# print('min value: ', m)
#
# # method2
# M, m = None, None
# for score in scores:
#     if M == None or score > M:
#         M = score
#     if m == None or score < m:
#         m = score
# print('Max value: ', M)
# print('min value: ', m)

# # 1-74) Min-max Normalization: 데이터들이 0~1 사이의 값을 가짐! (최소0, 최대1이 됨. 무조건.)
#
# scores = [-20, 60, 40, 70, 120]
# M, m = scores[0], scores[0]
#
# for score in scores:
#     if score > M:
#         M = score
#     if score < m:
#         m = score
# print('Max value: ', M)
# print('min value: ', m)
#
# for score_idx in range(len(scores)):
#     scores[score_idx] = (scores[score_idx] - m) / (M - m)
# print('scores after normalization:\n', scores)
#
# # 초기화
# M, m = scores[0], scores[0]
# for score in scores:
#     if score > M:
#         M = score
#     if score < m:
#         m = score
# print('Max value: ', M)
# print('min value: ', m)

# # 1-75) 최댓값, 최솟값 위치 구하기
#
# scores = [60, -20, 40, 120, 70]
# M, m = None, None
# M_idx, m_idx = 0, 0
#
# for score_idx in range(len(scores)):
#     score = scores[score_idx]
#
#     if M == None or score > M:
#         M = score
#         M_idx = score_idx
#     if m == None or score < m:
#         m = score
#         m_idx = score_idx
# print('M/M_idx: ', M, M_idx)
# print('m/m_idx: ', m, m_idx)

# # 1-76) Sorting
#
# scores = [40, 20 , 30, 10, 50]
# sorted_scores = list()
#
# for _ in range(len(scores)):
#     M, M_idx = scores[0], 0 # 맨 처음거로 그냥 초기화 해주고 아래 for문을 통해 얘랑 비교해서 최댓값 찾고, 그걸 sorted_scores에 넣어주고 scores에서는 지워준다. > 내림차순 완성!
#
#     for score_idx in range(len(scores)):
#         if scores[score_idx] > M:
#             M = scores[score_idx]
#             M_idx = score_idx
#
#     sorted_scores.append(M)
#     del scores[M_idx]
#
#     print('remaining scores: ', scores)
#     print('sorted scores: ', sorted_scores, '\n')

# # 1-77) Accuracy 구하기
#
# predictions = [0, 1, 0, 2, 1, 2, 0]
# labels = [1, 1, 0, 0, 1, 2, 1]
# n_correct = 0
#
# for pred_idx in range(len(predictions)):
#     if predictions[pred_idx] == labels[pred_idx]:
#         n_correct += 1
#
# # accuracy = 맞은 개수 / 데이터 수
# accuracy = n_correct / len(predictions)
# print('accuracy[%]: ', accuracy*100, '%') # %로 나타내려면 *100

# # 핸드폰 사진
# # 1-78) Confusion Vector 구하기 : confusion matrix라는게 있음(모델학습이 끝나고 나서 prediction값과, 정답과의 오차를 계산하기 위해서 만든 매트릭스! 2가지 타입으로 그릴 수 있다!)
# # 1) 이진분류해서 그릴 수 있음(spam/ham.. 암/암x... 불량품/정상품 등) >> 모델이 예측한 값이 있고, 실제값이 있음..
# # confusion matrix의 원리를 조금 이해하게 하기 위해 confusion vector를 그냥 해보신거임!

# predictions = [0, 1, 0, 2, 1, 2, 0]
# labels = [1, 1, 0, 0, 1, 2, 1]
#
# M_class = None
# for label in labels:
#     if M_class == None or label > M_class:
#         M_class = label
#         # print(M_class)
# M_class += 1
# # print(M_class)
#
# class_cnts, correct_cnts, confusion_vec = list(), list(), list()
# for _ in range(M_class):
#     class_cnts.append(0)
#     correct_cnts.append(0)
#     confusion_vec.append(None) # 왜 얘만 None? class_cnts, correct_cnts는 1씩 '더해주려고' 0을 append. None+1은 error뜸
#
# for pred_idx in range(len(predictions)):
#     pred = predictions[pred_idx]
#     label = labels[pred_idx]
#
#     class_cnts[label] += 1
#     if pred == label:
#         correct_cnts[label] += 1
# # print(class_cnts, correct_cnts, confusion_vec)
#
# # confusion vector: 클래스 당 맞는 개수 / 클래스 당 데이터 수
# for class_idx in range(M_class):
#     confusion_vec[class_idx] = correct_cnts[class_idx] / class_cnts[class_idx]
# print('confusion vector: ', confusion_vec)

# # 1-79) Histogram 구하기(막대그래프 - 핸드폰 사진)
#
# scores = [50, 20, 30, 40, 10, 50, 70, 80, 90, 20, 30]
# cutoffs = [0, 20, 40, 60, 80]
# histogram = [0, 0, 0, 0, 0]
#
# for score in scores:
#     if score > cutoffs[4]:
#         histogram[4] += 1
#     elif score > cutoffs[3]:
#         histogram[3] += 1
#     elif score > cutoffs[2]:
#         histogram[2] += 1
#     elif score > cutoffs[1]:
#         histogram[1] += 1
#     elif score > cutoffs[0]:
#         histogram[0] += 1
#     else:
#         pass
# print('histogram of the scores: ', histogram)


# # 1-80) 절댓값 구하기
#
# numbers = [-2, 2, -1, 3, -4, 9]
# abs_numbers = list()
#
# for num in numbers:
#     if num < 0:
#         abs_numbers.append(-num)
#     else:
#         abs_numbers.append(num)
# print(abs_numbers)

# # 1-81) Manhattan Distance
#
# v1 = [1,3,5,2,1,5,2]
# v2 = [2,3,1,5,2,1,3]
#
# m_distance = 0
#
# for dim_idx in range(len(v1)):
#     sub = v1[dim_idx] - v2[dim_idx]
#     if sub < 0:
#         m_distance += -sub
#     else:
#         m_distance += sub
# print('Manhattan distance: ', m_distance)

# # 1-82) Nested List(list in list) 만들기 & 원소 접근하기
# # list in list는 사실 행렬임! numpy 를 import해서 np.array(scores)하면
# # [[10 20 30]
# #   [50 60 370]] 이렇게 됨
#
# scores = [[10,20,30], [50,60,70]]
#
# print(scores)
# print(scores[0])
# print(scores[1])
# print(scores[0][0], scores[0][1], scores[0][2])
# print(scores[1][0], scores[1][1], scores[1][2])

# # 1-83) Nested List 원소 접근하기(2)
# scores = [[10,20,30], [50,60,70]]
#
# for student_scores in scores:
#     print(student_scores)
#     for score in student_scores:
#         print(score)

# # 1-84) 학생별 평균점수 구하기
#
# scores = [[10,15,20], [20,25,30], [30,35,40], [40,45,50]]
#
# n_class = len(scores[0])
# student_score_means = list()
#
# for student_scores in scores:
#     student_score_sum = 0
#     for score in student_scores:
#         student_score_sum += score
#     student_score_means.append(student_score_sum / n_class)
#
# print("mean of students' scores: ", student_score_means)

# # 1-85) 과목별 평균점수 구하기
#
# scores = [[10,15,20], [20,25,30], [30,35,40], [40,45,50]]
#
# n_student = len(scores)
# n_class = len(scores[0])
#
# class_score_sums = list()
# class_score_means = list()
#
# # set the sum of class scores as 0
# for _ in range(n_class):
#     class_score_sums.append(0) # 뭐를 계속 넣어줘야 한다! 그러면 0을 먼저 append해줘야 함! 0으로 초기화시켜줘야 더할 수 있음!
# # calculate the sum of class scores
# for student_scores in scores:
#     for class_idx in range(n_class):
#         class_score_sums[class_idx] += student_scores[class_idx]
# print("sum of classes' scores: ", class_score_sums)
#
# # calculate the mean of class scores
# for class_idx in range(n_class):
#     class_score_means.append(class_score_sums[class_idx] / n_student)
# print("mean of classes' scores: ", class_score_means)


# # 1-86) Mean Subtraction: 평균 0나옴!
#
# scores = [[10,15,20], [20,25,30], [30,35,40], [40,45,50]]
#
# n_student = len(scores)
# n_class = len(scores[0])
#
# class_score_sums = list()
# class_score_means = list()
#
# for _ in range(n_class):
#     class_score_sums.append(0)
# for student_scores in scores:
#     for class_idx in range(n_class):
#         class_score_sums[class_idx] += student_scores[class_idx]
# for class_idx in range(n_class):
#     class_score_means.append(class_score_sums[class_idx] / n_student)
# print("mean of classes' scores: ", class_score_means)
#
# for student_idx in range(n_student):
#     for class_idx in range(n_class):
#         scores[student_idx][class_idx] -= class_score_means[class_idx]
#
# # print(scores)
#
# # 초기화
# class_score_sums = list()
# class_score_means = list()
#
# for _ in range(n_class):
#     class_score_sums.append(0)
# for student_scores in scores:
#     for class_idx in range(n_class):
#         class_score_sums[class_idx] += student_scores[class_idx]
# for class_idx in range(n_class):
#     class_score_means.append(class_score_sums[class_idx] / n_student)
# print("mean of classes' scores: ", class_score_means)

# # 1-87) 분산과 표준편차(5)
#
# scores = [[10,15,20], [20,25,30], [30,35,40], [40,45,50]]
#
# n_student = len(scores)
# n_class = len(scores[0])
#
# class_score_sums = list()
# class_score_square_sums = list()
#
# class_score_variances = list()
# class_score_stds = list()
#
# # set the sum of class scores as 0
# for _ in range(n_class):
#     class_score_sums.append(0)
#     class_score_square_sums.append(0)
#
# # classes' sums, squared sums
# for student_scores in scores:
#     for class_idx in range(n_class):
#         class_score_sums[class_idx] += student_scores[class_idx]
#         class_score_square_sums[class_idx] += student_scores[class_idx]**2
# # print(class_score_sums)
# # print(class_score_square_sums)
#
# # classes' variance
# for class_idx in range(n_class):
#     mos = class_score_square_sums[class_idx] / n_student
#     som = (class_score_sums[class_idx] / n_student) **2
#
#     variance = mos - som
#     std = variance**0.5
#
#     class_score_variances.append(variance)
#     class_score_stds.append(std)
#
# # classes' variances, standard deviations
# print('variances: ', class_score_variances)
# print('standard deviations: ', class_score_stds)
#
#


# # 1-88) Standardization(5)
#
# scores = [[10,15,20], [20,25,30], [30,35,40], [40,45,50]]
#
# n_student = len(scores)
# n_class = len(scores[0])
#
# class_score_sums = list()
# class_score_square_sums = list()
#
# for _ in range(n_class):
#     class_score_sums.append(0)
#     class_score_square_sums.append(0)
# for student_scores in scores:
#     for class_idx in range(n_class):
#         class_score_sums[class_idx] += student_scores[class_idx]
#         class_score_square_sums[class_idx] += student_scores[class_idx]**2
#
# class_score_variance = list()
# class_score_stds = list()
#
# for class_idx in range(n_class):
#     mos = class_score_square_sums[class_idx] / n_student
#     som = (class_score_sums[class_idx] / n_student) ** 2
#
#     variance = mos - som
#     std = variance**0.5
#
#     class_score_variance.append(variance)
#     class_score_stds.append(std)
#
# print('standard deviations: ', class_score_stds)
#
# # standardization
# for student_idx in range(n_student):
#     for class_idx in range(n_class):
#         score = scores[student_idx][class_idx]
#         mean = class_score_sums[class_idx] / n_student
#         std = class_score_stds[class_idx]
#
#         scores[student_idx][class_idx] = (score - mean) / std
#
# # print(scores)
#
# class_score_sums = list()
# class_score_square_sums = list()
#
# for _ in range(n_class):
#     class_score_sums.append(0)
#     class_score_square_sums.append(0)
#
# for student_scores in scores:
#     for class_idx in range(n_class):
#         class_score_sums[class_idx] += student_scores[class_idx]
#         class_score_square_sums[class_idx] += student_scores[class_idx]**2
#
# class_score_variance = list()
# class_score_stds = list()
#
# for class_idx in range(n_class):
#     mos = class_score_square_sums[class_idx] / n_student
#     som = (class_score_sums[class_idx] / n_student)**2
#
#     variance = mos - som
#     std = variance**0.5
#
#     class_score_variance.append(variance)
#     class_score_stds.append(std)
#
# print('standard deviations: ', class_score_stds)


# # ---------2번째: class_score_means 변수 추가----------
#
# scores = [[10,15,20],
#           [20,25,30],
#           [30,35,40],
#           [40,45,50]]
#
# n_student = len(scores)
# n_class = len(scores[0])
#
# class_score_sums = list()
# class_score_square_sums = list()
# class_score_means = list()
#
# for _ in range(n_class):
#     class_score_sums.append(0)
#     class_score_square_sums.append(0)
#     class_score_means.append(None)
#
# for student_scores in scores:
#     for class_idx in range(n_class):
#         class_score_sums[class_idx] += student_scores[class_idx]
#         class_score_square_sums[class_idx] += student_scores[class_idx]**2
#
# # class_score_means
# for class_idx in range(n_class):
#     mean = class_score_sums[class_idx] / n_student
#     class_score_means[class_idx] = mean
# # print(class_score_means)
#
# class_score_variance = list()
# class_score_stds = list()
#
# for class_idx in range(n_class):
#     mos = class_score_square_sums[class_idx] / n_student
#     som = (class_score_sums[class_idx] / n_student)**2
#
#     variance = mos - som
#     std = variance**0.5
#
#     class_score_variance.append(variance)
#     class_score_stds.append(std)
#
# print('standard deviations: ', class_score_stds)
#
#
# # standardization
# for student_idx in range(n_student): # 번호만 필요하니까 idx로 / 사실 위에도 마찬가지임..
#     for class_idx in range(n_class):
#         score = scores[student_idx][class_idx]
#         mean = class_score_means[class_idx]
#         std = class_score_stds[class_idx]
#
#         scores[student_idx][class_idx] = (score - mean) / std # 업데이트!! 왼쪽에는 score 변수 쓰면x, 인덱싱 해줘야 함! / score는 이전값!
#
# print(scores)
#
# # 표준화가 잘 됐나 std구해보기!
#
# class_score_sums = list()
# class_score_square_sums = list()
# class_score_means = list()
#
# for _ in range(n_class):
#     class_score_sums.append(0)
#     class_score_square_sums.append(0)
#     class_score_means.append(None)
#
# for student_scores in scores:
#     for class_idx in range(n_class):
#         class_score_sums[class_idx] += student_scores[class_idx]
#         class_score_square_sums[class_idx] += student_scores[class_idx] ** 2
# for class_idx in range(n_class):
#     mean = class_score_sums[class_idx] / n_student
#     class_score_means[class_idx] = mean
# # print(class_score_means)
#
# class_score_variance = list()
# class_score_stds = list()
#
# for class_idx in range(n_class):
#     mos = class_score_square_sums[class_idx] / n_student
#     som = (class_score_sums[class_idx] / n_student) ** 2
#
#     variance = mos - som
#     std = variance ** 0.5
#
#     class_score_variance.append(variance)
#     class_score_stds.append(std)
#
# print('standard deviations: ', class_score_stds)
#
# # 1-89) Hadamard Product(5)
#
# # columnize로..열기준으로 데이터를 봄(이게 표준임. 열방향으로 1,2,3,4가 하나의 벡터임) > hadamar하면 행끼리 곱해주면 됨)
# vectors = [[1, 11, 21],
#            [2, 12, 22],
#            [3, 13, 23],
#            [4, 14, 24]]
#
# # 원소들 곱한 값
# h_prod = list()
# for dim_data in vectors:
#     dim_prod = 1 # 곱하기 해주기 위해서
#     for dim_val in dim_data:
#         dim_prod *= dim_val
#     h_prod.append(dim_prod)
#
# print(h_prod)
#
# # 1-90) Vector Norm(4)
#
# vectors = [[1, 11, 21],
#            [2, 12, 22],
#            [3, 13, 23],
#            [4, 14, 24]]
#
# n_vector = len(vectors[0])
# v_norms = list()
# for _ in range(n_vector):
#     v_norms.append(0)
# print(v_norms)
#
# for dim_data in vectors:
#     for dim_idx in range(n_vector): # index값만 필요하면 이렇게 idx로 해주기!
#         v_norms[dim_idx] += dim_data[dim_idx]**2
# print(v_norms)
#
# for vec_idx in range(n_vector):
#     v_norms[vec_idx] **= 0.5
# print(v_norms)
#
# # 1-91) Making Unit Vectors(4)
# vectors = [[1, 11, 21],
#            [2, 12, 22],
#            [3, 13, 23],
#            [4, 14, 24]]
#
# n_dim = len(vectors)
# n_vector = len(vectors[0])
#
# # vector norm
# v_norms = list()
# for _ in range(n_vector):
#     v_norms.append(0)
#
# for dim_data in vectors:
#     for dim_idx in range(n_vector):
#         v_norms[dim_idx] += dim_data[dim_idx]**2
#
# for vec_idx in range(n_vector):
#     v_norms[vec_idx] **= 0.5
# print(v_norms)
#
# # unit vector
# for dim_idx in range(n_dim):
#     for vec_idx in range(n_vector):
#         vectors[dim_idx][vec_idx] /= v_norms[vec_idx]
#
# # unit vector norm
# v_norms = list()
# for _ in range(n_vector):
#     v_norms.append(0)
#
# for dim_data in vectors:
#     for dim_idx in range(n_vector):
#         v_norms[dim_idx] += dim_data[dim_idx]**2
#
# for vec_idx in range(n_vector):
#     v_norms[vec_idx] **= 0.5
# print(v_norms)

# # 1-92)
#
# vectors = [[1, 11],
#            [2, 12],
#            [3, 13],
#            [4, 14]]
#
# d_prod = 0
# for dim_data in vectors:
#     d_prod += dim_data[0]*dim_data[1]
# print('dot product: ', d_prod)

# # 1-93) Euclidean Distance(4) - 차의 제곱 합에 루트!!!! 거리구하는거야아아!!!!
#
# vectors = [[1, 11],
#            [2, 12],
#            [3, 13],
#            [4, 14]]
#
# e_distance = 0
#
# for dim_data in vectors:
#     diff = dim_data[0] - dim_data[1]
#     diff_square = diff**2
#
#     e_distance += diff_square
#
# e_distance **= 0.5
# print('Euclidean distance: ', e_distance)

# # 1-94) 과목별 최고점, 최우수 학생 구하기
#
# scores = [[10,40,20],
#           [50,20,60],
#           [70,40,30],
#           [30,80,40]]
#
# n_student = len(scores)
# n_class = len(scores[0])
#
# M_classes = scores[0] # 기준을 첫 학생 점수로! 초기화
# M_idx_classes = list()
# for _ in range(n_class):
#     M_idx_classes.append(0)
#
# for student_idx in range(n_student):
#     student_scores = scores[student_idx]
#
#     for class_idx in range(n_class):
#         score = student_scores[class_idx]
#         if score > M_classes[class_idx]:
#             M_classes[class_idx] = score
#             M_idx_classes[class_idx] = student_idx
# print('Max scores: ', M_classes)
# print('Max score indices: ', M_idx_classes)

# # 1-95) One-hot Encoding
# # one-hot encodingL: labels는 4가지 class가 있음. class마다 index를 할당해주는것! 0 ->[0], 1 ->[1]....0번이라는 데이터를 아래와 같은 벡터에 표현해줌.
# # 그래서 one-hot encoding임! 하나로만 표현!
# # 자연어 처리에서 많이 쓰임! ex. I love you. 라는 말을 컴퓨터가 인식하게끔 해주는 것.(10000개의 단어가 있다면.. i는 100, love는 500, you는 300번째라면 그 때만 1이 나타나는 벡터로 나타낼 수 있음)
# # 왜 필요함? I love 뒤에는 어떤 대상..이 오는데 컴퓨터에 이 자리에 뭐가 오는지.. 확률적으로 계산해서 학습시킬 수 있음 / I 다음에 you가 바로 올 확률은 적구나..등
#
# labels = [0, 1, 2, 1, 0, 3]
#
# n_label = len(labels)
# n_class = 0
# for label in labels:
#     if label > n_class:
#         n_class = label
# n_class += 1
#
# one_hot_mat = list()
#
# for label in labels:
#     one_hot_vec = list()
#     for _ in range(n_class):
#         one_hot_vec.append(0)
#     one_hot_vec[label] = 1
#
#     one_hot_mat.append(one_hot_vec)
# print(one_hot_mat)

# 1-96) Accuracy 구하기(2)

predictions = [[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0], [1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]]

labels = [0,1,2,1,0,3]

# labels > one_hot_mat (one-hot encoding)
n_class = 0
for label in labels:
    if label > n_class:
        n_class = label
n_class += 1

one_hot_mat = list()
for label in labels:
    one_hot_vec = list()
    for _ in range(n_class):
        one_hot_vec.append(0)
    one_hot_vec[label] = 1
    one_hot_mat.append(one_hot_vec)
# print(one_hot_mat)

n_pred = len(predictions)
n_class = len(predictions[0])

accuracy = 0
for pred_idx in range(n_pred):
    prediction = predictions[pred_idx]
    label = one_hot_mat[pred_idx]

    correct_cnt = 0
    for class_idx in range(n_class):
        if prediction[class_idx] == label[class_idx]:
            correct_cnt += 1

    if correct_cnt == n_class: # 4랑 같으면?!음....???
        accuracy += 1

accuracy /= n_pred
print('accuracy: ', accuracy)













