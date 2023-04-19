# # 1-48) Mean Subtraction(2)
#
# math_scores = [40, 60, 80]
# english_scores = [30, 40, 50]
#
# n_class = 2
# n_student = len(math_scores)
#
# score_sums = list()
# score_means = list()
#
# for _ in range(n_class):
#     score_sums.append(0)
#
# for student_idx in range(n_student):
#     score_sums[0] += math_scores[student_idx] # 더하려면 먼저 어떤 값이 있어야 함. 그래서 위에서 0으로 채워준거임.(append(0))
#     score_sums[1] += english_scores[student_idx]
#
# for class_idx in range(n_class):
#     class_mean = score_sums[class_idx] / n_student
#     score_means.append(class_mean) # 얜 append만 하면되니까!(+ 연산 필요x)
#
# for student_idx in range(n_student):
#     math_scores[student_idx] -= score_means[0]
#     english_scores[student_idx] -= score_means[1]
#
# print('Math scores after mean subtraction: ', math_scores)
# print('English scores after mean subtraction: ', english_scores)

# # 1-49) 분산과 표준편차(3)
#
# scores = [10, 20, 30]
# n_student = len(scores)
# score_sum, score_square_sum = 0, 0
#
# for score in scores:
#     score_sum += score
#     score_square_sum += score**2
#
# mean = score_sum / n_student
# square_of_mean = mean**2
# mean_of_square = score_square_sum / n_student
#
# variance = mean_of_square - square_of_mean
# std = variance**0.5
#
# print('variance: ', variance)
# print('standard deviation: ', std)

# # 1-50) standardization(3) : 평균0, 분산&표준편차1 > 표준정규분포를 따른다!
#
# scores = [10, 20, 30]
# n_student = len(scores)
# score_sum, score_square_sum = 0,0
#
# for score in scores:
#     score_sum += score
#     score_square_sum += score**2
#
# mean = score_sum / n_student
# mean_of_square = score_square_sum / n_student
# square_of_mean = mean**2
#
# variance = mean_of_square - square_of_mean
# std = variance**0.5
#
# for student_idx in range(n_student):
#     scores[student_idx] = (scores[student_idx] - mean) / std
# print(scores)
#
# score_sum, score_square_sum = 0,0 # 다시 0으로 초기화 해줘야 함!
#
# for score in scores:
#     score_sum += score
#     score_square_sum += score**2
#
# mean = score_sum / n_student
# mean_of_square = score_square_sum / n_student
# square_of_mean = mean**2
#
# variance = mean_of_square - square_of_mean
# std = variance**0.5
#
# print('mean: ', mean)
# print('standard deviation: ', std)

# # 1-51) 분산과 표준편차(4)
#
# math_scores, english_scores = [50,60,70], [30,40,50]
# n_student = len(math_scores)
#
# # 합, 제곱합
# math_sum, english_sum = 0, 0
# math_square_sum, english_square_sum = 0,0
#
# for student_idx in range(n_student):
#     math_sum += math_scores[student_idx]
#     math_square_sum += math_scores[student_idx]**2
#
#     english_sum += english_scores[student_idx]
#     english_square_sum += english_scores[student_idx]**2
# # 평균
# math_mean = math_sum / n_student
# english_mean = english_sum /n_student
# # 분산
# math_variance = math_square_sum/n_student - math_mean**2
# english_variance = english_square_sum/n_student - english_mean**2
# # 표준편차
# math_std = math_variance**0.5
# english_std = english_variance**0.5
#
# print('mean/std of Math: ', math_mean, math_std)
# print('mean/std of English: ', english_mean, english_std)

# # 1-52) Standardization(4)
#
# math_scores, english_scores = [50,60,70], [30,40,50]
# n_student = len(math_scores)
# math_sum, english_sum = 0,0
# math_square_sum, english_square_sum = 0,0
#
# for student_idx in range(n_student):
#     math_sum += math_scores[student_idx]
#     english_sum += english_scores[student_idx]
#
#     math_square_sum += math_scores[student_idx]**2
#     english_square_sum += english_scores[student_idx]**2
#
# math_mean = math_sum / n_student
# english_mean = english_sum / n_student
#
# math_variance = math_square_sum/n_student - math_mean**2
# english_variance = english_square_sum/n_student - english_mean**2
#
# math_std = math_variance**0.5
# english_std = english_variance**0.5
#
# # Standardization
# for student_idx in range(n_student):
#     math_scores[student_idx] = (math_scores[student_idx] - math_mean) / math_std
#     english_scores[student_idx] = (english_scores[student_idx] - english_mean) / english_std
#
# print('Math scores after standardization: ', math_scores)
# print('English scores after standardization: ', english_scores)
#
# # 초기화
# math_sum, english_sum = 0,0
# math_square_sum, english_square_sum = 0,0
#
# for student_idx in range(n_student):
#     math_sum += math_scores[student_idx]
#     english_sum += english_scores[student_idx]
#
#     math_square_sum += math_scores[student_idx]**2
#     english_square_sum += english_scores[student_idx]**2
#
# math_mean = math_sum / n_student
# english_mean = english_sum / n_student
#
# math_variance = math_square_sum/n_student - math_mean**2
# english_variance = english_square_sum/n_student - english_mean**2
#
# math_std = math_variance**0.5
# english_std = english_variance**0.5
#
# print('mean/std of Math: ', math_mean, math_std)
# print('mean/std of English: ', english_mean, english_std)

# # 1-53) Hadamard Product(4)
#
# v1 = [1,2,3,4,5]
# v2 = [10,20,30,40,50]
#
# # method1
# v3 =list()
# for dim_idx in range(len(v1)):
#     v3.append(v1[dim_idx] * v2[dim_idx])
# print(v3)
#
# # method2
# v3 = list()
# for _ in range(len(v1)):
#     v3.append(0)
#
# for dim_idx in range(len(v1)):
#     v3[dim_idx] = v1[dim_idx]*v2[dim_idx]
# print(v3)

# # 1-54) Vector Norm(3)
#
# v1 = [1,2,3]
#
# square_sum = 0
# for dim_val in v1:
#     square_sum += dim_val**2
# norm = square_sum**0.5
# print('norm of v1: ', norm)

# # 1-55) Making Unit Vector
#
# v1 = [1,2,3]
#
# square_sum = 0
# for dim_val in v1:
#     square_sum += dim_val**2
# norm = square_sum**0.5
# print('norm of v1: ', norm)
#
# # unit vector 만들기 (단위 벡터: 크기가 1인 벡터) - norm 필요
# for dim_idx in range(len(v1)):
#     v1[dim_idx] /= norm
#
# # 초기화
# square_sum = 0
# for dim_val in v1:
#     square_sum += dim_val**2
# norm = square_sum**0.5
# print('norm of v1: ', norm)

# 1-56) Dot Product(3)

# v1, v2 = [1,2,3], [3,4,5]
#
# dot_prod = 0
# for dim_idx in range(len(v1)):
#     dot_prod += v1[dim_idx] * v2[dim_idx]
# print('dot product of v1 and v2: ', dot_prod)

# # 1-57) Euclidean Distance(3) : 차의 제곱 합에 루트
#
# v1, v2 = [1,2,3], [3,4,5]
#
# diff_square_sum = 0
# for dim_idx in range(len(v1)):
#     diff_square_sum += (v1[dim_idx] - v2[dim_idx])**2
# e_distance = diff_square_sum**0.5
#
# print('Euclidean distance between v1 and v2: ', e_distance)

# # 1-58) Mean Squared Error(3) : error니까 예측값과 나온 값,데이터의 차이. 이 차의 제곱 합을 평균내준게 mse!
#
# predictions = [10,20,30]
# labels = [10,25,40]
#
# n_data = len(predictions)
# diff_square_sum = 0
#
# for data_idx in range(n_data):
#     diff_square_sum += (predictions[data_idx] - labels[data_idx])**2
# mse = diff_square_sum / n_data
# print('MSE: ', mse)

# # 1-60) 숫자 빈도 구하기 - 0,1,2,3,4가 number_cnt의 인덱스가 되는거임!
#
# numbers = [0,2,4,2,1,4,3,1,2,3,4,1,2,3,4]
# number_cnt = [0,0,0,0,0]
#
# for num in numbers:
#     number_cnt[num] += 1 # num이 1이면 number_cnt의 1번 인덱스에 +1이 됨
# print(number_cnt)

# # 1-60) 합격 알려주기
#
# score = 60
#
# if score > 50:
#     print('Pass!')

# # 1-61) 합격/불합격 알려주기
#
# score = 40
# cutoff = 50
#
# if score > cutoff:
#     print('Pass!')
# else:
#     print('Try Again!')

# # 1-62) 초를 분초로 표현하기
#
# seconds = 200
#
# if seconds >= 60:
#     minutes = seconds // 60 # 몫
#     seconds -= minutes*60 # seconds = seconds % 60 / seconds %= 60 > 이렇게도 O
# else:
#     minutes = 0
#
# print(minutes, 'min', seconds, 'sec')

# # 1-63) 초를 시분초로 표현하기
#
# seconds = 5000
#
# if seconds >= 3600:
#     hours = seconds // 3600
#     seconds -= hours*3600
# else:
#     hours = 0
#
# if seconds>= 60:
#     minutes = seconds // 60
#     seconds -= minutes*60
# else:
#     minutes = 0
# print(hours, 'hr', minutes, 'min', seconds, 'sec')

# # 1-64) 홀수/짝수 구하기
#
# number = 10
#
# if number % 2 == 0:
#     print('Even!')
# else:
#     print('Odd!')

# # 1-65) 두 수 비교하기
#
# num1, num2 = 10, 10
#
# if num1 > num2:
#     print('first number')
# elif num1 == num2:
#     print('equal')
# else:
#     print('second number')

# # 1-66) 성적을 평점으로 바꾸기
#
# score = 70
#
# if score > 80:
#     grade = 'A'
# elif score > 60:
#     grade = 'B'
# elif score > 40:
#     grade = 'C'
# else:
#     grade = 'F'
# print('Grade: ', grade)

# # 1-67) 합격/불합격 알려주기(2)
#
# scores = [20, 50, 10, 60, 70]
# cutoff = 50
#
# for score in scores:
#     if score > cutoff:
#         print('Pass!')
#     else:
#         print('Try Again!')

# # 1-68) 성정을 평점으로 바꾸기(2)
#
# scores = [20, 50, 10, 60, 90]
# grades = list()
#
# for score in scores:
#     if score > 80:
#         grades.append('A')
#     elif score > 60:
#         grades.append('B')
#     elif score > 40:
#         grades.append('C')
#     else:
#         grades.append('F')
# print(grades)

# 1-69) 합격/불합격 학생들의 평균 구하기

scores = [20, 50, 10, 60, 90]
cutoff = 50

p_score_sum, n_p = 0, 0
np_score_sum, n_np = 0, 0

for score in scores:
    if score > cutoff:
        p_score_sum += score
        n_p += 1
    else:
        np_score_sum += score
        n_np += 1

p_score_mean = p_score_sum / n_p
np_score_mean = np_score_sum / n_np

print('mean of passed scores: ', p_score_mean)
print('mean of no passed scores: ', np_score_mean)
















