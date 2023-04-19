# # 1-4) 사칙연산 결과를 할당하고 출력하기
#
# int1, int2 = 4, 2
#
# add, sub = int1 + int2, int1 - int2
# mul, div = int1 * int2, int1 / int2
# quo, rem = int1 // int2, int1 % int2
# pow = int1 ** int2
#
# print(add, sub, mul, div, quo, rem, pow)

# # 1-5) 홀수 짝수 구하기
#
# int1, int2, int3, int4 = 1, 2, 3, 4
# div = 2
# rem1, rem2, rem3, rem4 = int1 % div, int2 % div, int3 % div, int4 % div
# print(rem1, rem2, rem3, rem4)
#
# int1, int2, int3, int4 = 1, 2, 3, 4
# div = 2
# rem1, rem2, rem3, rem4 = int1 % div, int2 % div, int3 % div, int4 % div
# print(rem1, rem2, rem3, rem4)

# # 1-6) 자기 자신에 연산하기
#
# score = 10
# print(score)
#
# score += 10
# print(score)
#
# score -= 2
# print(score)
#
# score *= 5
# print(score)
#
# score /= 5
# print(score)
#
# score **= 2
# print(score)

# # 1-7) 수학 점수의 합 구하기
#
# score1 = 10
# score2 = 20
# score3 = 30
#
# score_sum = score1 + score2 + score3
# print(score_sum)
#
# score1, score2, score3 = 10, 20,30
# score_sum = score1 + score2 + score3
# print(score_sum)

# # 1-8) 수학점수의 평균 구하기
#
# score1 = 10
# score2 = 20
# score3 = 30
# n_student = 3
#
# score_mean = (score1 + score2 + score3) / n_student
# print(score_mean)
#
# # 보통 이렇게 변수에 할당에서 많이 사용! 그래야 나중에 코드 수정할 때, 그 변수 찾아서 그 값만 바꿔주면 됨!
#
# score1, score2, score3 = 10, 20, 30
# n_student = 3
# score_mean = (score1 + score2 + score3) / n_student
# print(score_mean)

# # 1-9) 추가점수를 받은 학생들의 평균
#
# socre1 = 10
# score2 = 20
# score3 = 30
# n_student = 3
#
# mean = (score1 + score2 + score3) / n_student
# print(mean)
#
# # 추가점수
# score1 += 10
# score2 += 10
# score3 += 10
# print(score1, score2, score3)
#
# mean = (score1 + score2 + score3) / n_student
# print(mean)

# # 1-9) Mean Subtraction : 평균이 원점으로 이동한다!(비교용이) > 평균이 0
#
# score1 = 10
# score2 = 20
# score3 = 30
# n_student = 3
#
# score_mean = (score1 + score2 + score3) / n_student
#
# score1 -= score_mean
# score2 -= score_mean
# score3 -= score_mean
#
# score_mean = (score1 + score2 + score3) / n_student # (-10 + 0 + 10) / 3
# print(score_mean)

# # 1-11) 평균의 제곱과 제곱의 평균 : 분산!
#
# score1 = 10
# score2 = 20
# score3 = 30
# n_student = 3
#
# mean = (score1+score2+score3) / n_student
# square_of_mean = mean**2
# mean_of_square = (score1**2 + score2**2 + score3**2) / n_student
#
# print("square of mean: ", square_of_mean)
# print("mean of square: ", mean_of_square)

# # 1-12) 분산과 표준편차(variance, standar... 분산에 루트 씌운거=표준편차)
#
# score1 = 10
# score2 = 20
# score3 = 30
# n_student = 3
#
# score_mean = (score1 + score2 + score3) / n_student
# square_of_mean = score_mean**2
# mean_of_square = (score1**2 + score2**2 + score3**2) / n_student
#
# # 분산
# score_variance = mean_of_square - square_of_mean
# # 표준편차
# score_std = score_variance ** 0.5
#
# print('mean: ', score_mean)
# print('variance: ', score_variance)
# print('standard deviation: ', score_std)
#
# score1 = 10
# score2 = 20
# score3 = 30
# n_student = 3
#
# score_mean = (score1 + score2 + score3) / n_student
# square_of_mean = score_mean**2
# mean_of_square = (score1**2 + score2**2 + score3**2) / n_student
#
# score_variance = mean_of_square - square_of_mean
# score_std = score_variance**0.5
#
# print('mean: ', score_mean)
# print('variance: ', score_variance)
# print('standard deviation: ', score_std)
#
# score1 = 10
# score2 = 20
# score3 = 30
# n_student = 3
#
# score_mean = (score1 + score2 + score3) / n_student
# square_of_mean = score_mean**2
# mean_of_square = (score1**2 + score2**2 + score3**2) / n_student
#
# score_variance = mean_of_square - square_of_mean
# score_std = score_variance**0.5
#
# print('mean: ', score_mean)
# print('variance: ', score_variance)
# print('standard deviation: ', score_std)

# # # 1-13) standardization : 퍙균빼고, 표준편차로 나눠준다! >> 평균0, 표준편차1, 분산1로 됨! ex. 토익990 / 시험100 > 데이터마다 '스케일'이 다르다. 이걸 조정해줄때 standardization기법을 사용!
#
# score1 = 10
# score2 = 20
# score3 = 30
# n_student = 3
#
# score_mean = (score1 + score2 + score3) / n_student
# square_of_mean = score_mean**2
# mean_of_square = (score1**2 + score2**2 + score3**2) / n_student
# score_variance = mean_of_square - square_of_mean
# score_std = score_variance**0.5
#
# print('mean: ', score_mean)
# print('standard deviation: ', score_std)
#
# # standardization
# score1 = (score1 - score_mean) / score_std
# score2 = (score2 - score_mean) / score_std
# score3 = (score3 - score_mean) / score_std
#
# score_mean = (score1 + score2 + score3) / n_student
# square_of_mean = score_mean**2
# mean_of_square = (score1**2 + score2**2 + score3**2) / n_student
# score_variance = mean_of_square - square_of_mean
# score_std = score_variance**0.5
#
# print('mean: ', score_mean)
# print('standard deviation: ', score_std)


# 1-14) Vector-Vector Operations : 아래와 같은 vector연산을 자동으로 해주는게 numpy라이브러리!

# x1, y1, z1 = 1, 2, 3 # v1(벡터)
# x2, y2, z2 = 3, 4, 5 # v2
#
# x3, y3, z3 = x1 + x2, y1 + y2, z1 + z2
# x4, y4, z4 = x1 - x2, y1 - y2, z1 - z2
# x5, y5, z5 = x1 * x2, y1 * y2, z1 * z2
#
# print(x3, y3, z3)
# print(x4, y4, z4)
# print(x5, y5, z5)


# # 1-15) Scalar-Vector Operations: 스칼라와 벡터 연산 가능!

# a = 10
# x1, y1, z1 = 1, 2, 3
#
# x2, y2, z2 = a*x1, a*y1, a*z1
# x3, y3, z3 = a+x1, a+y1, a+z1
# x4, y4, z4 = a-x1, a-y1, a-z1


# 1-16) Vector Norm: 벡터 크기 구하기!

# x, y, z = 1, 2, 3
#
# norm = (x**2 + y**2 + z**2)**0.5
# print(norm)

# 1-17) Making Unit Vectors: 벡터 norm을 이용해서 unit vector만들기! 크기가 1인 벡터.. 유닛 벡터..(단일벡터..)
# >> 벡터를 norm으로 나눴을때 단일 벡터가 되는거.증명하는거 숙제!!!

x, y, z = 1, 2, 3
norm = (x**2 + y**2 + z**2)**0.5
print(norm)

x, y, z = x/norm, y/norm, z/norm
norm = (x**2 + y**2 + z**2)**0.5
print(norm)

# 1-18) Dot Product: 벡터가 2개가 있음. 각각 원소를 곱해서 얘네를 다 더해주면 두 벡터의 내적임!

# 1- 19) Euclidean distance: 피타고라스 정리 이용해서 '거리' 구하는 것

# 1-20) squared error: 햇(^) 이걸 쓰면 예측값임. 우리가 예측한 값과 정답 값 사이의 error 를 구하는 것!!! 오차!!!!!!!! 한 데이터의.


# ---------------------------------------------------------------------------------------------------------------




