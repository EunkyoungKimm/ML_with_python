# # 1-21) Mean Squared Error : error계산할때 MSE 많이 쓰임! (에러 제곱합 평균)
#
# pred1, pred2, pred3 = 10, 20, 30
# y1, y2, y3 = 10, 25, 40
# n_data = 3
#
# s_error1 = (pred1 - y1)**2
# s_error2 = (pred2 - y2)**2
# s_error3 = (pred3 - y3)**2
#
# mse = (s_error1 + s_error2 + s_error3) / n_data
# print(mse)

# # 1-21) List 만들기 & 원소 접근하기
#
# scores = [10,20,30]
# print(scores[0])
# print(scores[1])
# print(scores[2])

# # 1-22) List의 원소 수정하기: 리스트는 인덱싱이 가능 >> 인덱스로 직접접근해 수정해줌!
#
# scores = [10, 20, 30]
# print(scores)
#
# scores[0] = 100
# scores[1] = 200
# print(scores)

# # 1-24) 수학점수들의 평균 구하기(2)
#
# scores = [10,20,30]
# n_student = len(scores) # 3 > len함수: 리스트 원소 개수 알 수O
#
# mean = (scores[0] + scores[1] + scores[2]) / n_student
# print('score mean: ', mean)

# sum은 이미 내장함수. 그래서 변수명을 사용할때 sum_과 같이 underscore 이용해서 오류를 피함!(충돌 피함)

# # 1-25) Mean Subtraction(2) : mean subtraction 해주면 data들의 평균은 0이 된다 ( 평균뺀거 합 평균=0)
#
# scores = [10,20,30]
# n_student = len(scores)
#
# mean = (scores[0] + scores[1] + scores[2]) / n_student
# print('score mean: ', mean)
#
# # mean subtraction
# scores[0] -= mean
# scores[1] -= mean
# scores[2] -= mean
#
# mean = (scores[0] + scores[1] + scores[2]) / n_student
# print('score mean: ', mean)

# # 1-26) 분산과 표준편차(2): list 이용해서 분산과 표준편차 구하기 ( 제평-평제, 루트 제평-평제)
#
# scores = [10, 20, 30]
# n_student = len(scores)
#
# mean = (scores[0] + scores[1] + scores[2]) / n_student
# square_of_mean = mean**2 # 평균의 제곱
# mean_of_square = (scores[0]**2 + scores[1]**2 + scores[2]**2) / n_student # 제곱의 평균
#
# # 분산 = variance
# variance = mean_of_square - square_of_mean # MOS - SOM
# # 표준편차 = standard deviation = std
# std = variance**0.5 # square root of the variance
#
# print('score mean: ', mean)
# print('score standard deviation: ', std)

# # data scale조정 = data scaling하는거임/ 그중 하나가 z-score normalization (=standardization)
# # 1-27) Standardization(2) : standardizaion 후에 평균0, 표준편차1(분산1)로 바뀜 (표준정규분포!!)
# # (핸드폰 사진 : 구간별로 확인함. -1과 1사이, 이 구간에는 데이터의 68%정도가 들어감) (-2~2 사이에는 95%정도) (-3부터 3사이에는 99%)
# # (얘네들의 범위를 벗어났을때, 얘네들을 outlier라고 해줌)(우리학습에는 필요없는 애들!) (우리의 대표적인 경향을 따르지 않는 데이터들!)
# # min-max normalization : 분모에는 (데이터의 최대값-최소값), 분자에는 (데이터값-데이터최소값) >> 이 처리를 해주면 모든 데이터 값이 0~1값
# # ex. 토익 100 != 학교시험 100 (숫자는 같지만 의미 완전 다름) > 이걸 0~1 값으로 바꿔줘서 990점에서 100점의미와.. 그 의미를 알게 됨!
# # 이게 데이터 스케일링을 해준다는 것!(그 기법에 z-score normalization, min-max normalization 등이 있다는 것!)
#
# scores = [10, 20, 30]
# n_student = len(scores)
#
# # 1 -26 참고
# mean = (scores[0] + scores[1] + scores[2]) / n_student
# square_of_mean = mean**2
# mean_of_square = (scores[0]**2 + scores[1]**2 + scores[2]**2) / n_student
#
# variance = mean_of_square - square_of_mean
# std = variance**0.5
#
# print('score mean: ', mean)
# print('score standard deviation: ', std)
#
# scores[0] = (scores[0] - mean) / std
# scores[1] = (scores[1] - mean) / std
# scores[2] = (scores[2] - mean) / std
#
# # 1-26 참고
# mean = (scores[0] + scores[1] + scores[2]) / n_student
# square_of_mean = mean**2
# mean_of_square = (scores[0]**2 + scores[1]**2 + scores[2]**2) / n_student
#
# variance = mean_of_square - square_of_mean
# std = variance**0.5
#
# print('score mean: ', mean)
# print('score standard deviation: ', std)

# # 1-28) Hadamard Product(2) : 벡터 연산.. dot product는 '벡터의 내적'(ex. v1벡터가 a1,b1 이라는 원소를 가지고 있고.. 즉, 내적 값을 만듦(얘는 곱한걸 더해준 값))
# # 하다마드는 또 다른 새로운 형태의 벡터를 만드는 것! > numpy로 벡터 계산하면 원소별로 계산해줌. numpy로 곱하기 한걸 hadamard product!!
# # anaconda prompt에서 numpy로 하다마드 프로덕트 구한거 핸드폰 사진에 있음
# # 즉, numpy에서 곱하기한게 hadamard product임! (머신러닝, 딥러닝에서 진짜 많이 쓰이는 개념! LSTM(딥러닝 네트워크 중 하나) 여기 안에서도 하다마드가 쓰임)
# # 내적이랑 다름!!!!
# # 아다마르, 아다마드, 하다마드.. 여러가지로 불림!
#
# # method1
#
# v1, v2 = [1,2,3], [3,4,5]
# v3 = [v1[0]*v2[0], v1[1]*v2[1], v1[2]*v2[2]]
# print(v3)
#
# # methodd2
#
# v1, v2 = [1,2,3], [3,4,5]
# v3 = [0, 0, 0] # 공간을 만들어준거같음!
#
# v3[0] = v1[0]*v2[0]
# v3[1] = v1[1]*v2[1]
# v3[2] = v1[2]*v2[2]
# print(v3)

# # 1-29) lis() 와 append() : 리스트에 원소 추가
#
# v1 = list()
# print(v1)
#
# v1.append(1)
# print(v1)
#
# v1.append(2)
# print(v1)
#
# v1.append(3)
# print(v1)

# # 1-30) Hadamard Product(3)
#
# v1, v2 = [1,2,3], [3,4,5]
# v3 = list()
#
# v3.append(v1[0] * v2[0])
# v3.append(v1[1] * v2[1])
# v3.append(v1[2] * v2[2])
# print(v3)

# # 1-31) Vector Norm(2)
#
# v1 = [1,2,3]
#
# # method1
# norm = (v1[0]**2 + v1[1]**2 + v1[2]**2)**0.5
# print(norm)
#
# # method2
# norm = 0
# norm += v1[0]**2
# norm += v1[1]**2
# norm += v1[2]**2
# norm **= 0.5
# print(norm)

# # 1-32) Making Unit Vector(2) : vector norm이 1이 됨 ( 벡터norm은 벡터의 원소제곱합에 루트 > 그 norm으로 원소를 나눈게 unit vector, 그 벡터의 norm은 1임)
#
# v1 = [1,2,3]
#
# norm = (v1[0]**2 + v1[1]**2 + v1[2]**2)**0.5
# print(norm)
#
# v1 = [v1[0]/norm, v1[1]/norm, v1[2]/norm]
# norm = (v1[0]**2 + v1[1]**2 + v1[2]**2)**0.5
# print(norm)

# # 1-33) Dot Product(2) : 벡터의 내적 ( dot product, 내적은 원소들간 곱한거의 합)
# # 벡터라는 것은.. 학생들 별로 과목별 시험점수..
# # s1 = (국, 영, 수) 각 학생들마다 점수를 벡터 형식으로 나타낼 수 있음! 내적은 총점수를 안다거나 과목별로 점수를 정리해서..이럴때 내적을 사용할 수 있음!
# # 데이터를 정리하는 형식으로 벡터 형식으로 많이 정리함! 그래서 벡터 관련된 연산들을 잘 알고 있으면 좋음!
# # 내적이 제일 많이 쓰이는 데는 '유사도'측정할때라고 함..(유사도는 세타각에 영향을 많이 받음.. 벡터는 방향성을 띔. 내적값은 각 벡터의 놈에 영향을 받음! 벡터 크기가 크면 cos세타가 작아도.. 내적값이 큼. 벡터의 놈에 영향을 cos보다 더 많이 받음)
# # -1~1(음수일수록 유사도가 작다, 1일수록 유사도가 크다!)
# # 내적값이 크면 유사도가 큼(근데 이 내적값은 벡터norm에 영향을 많이 받아서, 내적하기전에 unit vector로 만들어주고 내적하면 -1~1사이 값이 나옴)
#
# v1, v2 = [1,2,3], [3,4,5]
#
# # method1
# dot_prod = v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]
# print(dot_prod)
#
# # method2
# dot_prod = 0
# dot_prod += v1[0]*v2[0]
# dot_prod += v1[1]*v2[1]
# dot_prod += v1[2]*v2[2]
# print(dot_prod)

# # 1-34) Euclidean Distance(2) : 원소끼리 뺀값을 제곱하고 루트(피타고라스 정리 생각하면 됨)
#
# v1, v2 = [1,2,3], [3,4,5]
#
# e_distance = 0
# e_distance += (v1[0] - v2[0])**2
# e_distance += (v1[1] - v2[1])**2
# e_distance += (v1[2] - v2[2])**2
# e_distance **= 0.5
# print(e_distance)

# # 1-35) Mean Squared Error(2)
#
# predictions = [10,20,30]
# labels = [10,25,40]
# n_data = len(predictions)
#
# mse = 0
# mse += (predictions[0] - labels[0])**2
# mse += (predictions[1] - labels[1])**2
# mse += (predictions[2] - labels[2])**2
# mse /= n_data
# print(mse)

# # 1-36) for loop으로 list의 원소 접근하기
#
# scores = [10,20,30]
#
# for score in scores:
#     prinT(score)

# # 1-37) List 원소들의 합 구하기
#
# scores = [10,20,30]
#
# score_sum = 0
# for score in scores:
#     score_sum += score
# print(score_sum)

# # 1-38) iteration 횟수 구하기 : for문이 몇번 도는지, 반복횟수 알고싶음
#
# numbers = [1,4,5,6,4,2,1]
# iter_cnt = 0
#
# for _ in numbers: # 우리한테 별의미없는 변수쓸때 그냥 _ 이거 씀(코드가 길어지면 어떤 변수명이 중요한지 아닌지 알 수 있음. 이건 그냥 횟수 세는거니까 _)
#     iter_cnt += 1
# print(iter_cnt)

# # 1-39) 1부터 100까지의 합 구하기 : 식 기억하기!
#
# num_sum = 0
# for i in range(101):
#     num_sum += i
# print(num_sum)

# python에 mean이라는 함수x, numpy에는 있음. np.mean, pd.mean은 있음!(np>name space) 그 안에서만 조심하면 됨. sum은 python의 내장함수라서 조심해야 함.


# # 1-40) 1부터 100까지 list 만들기
#
# numbers = list()
# for i in range(101):
#     numbers.append(i)
# print(numbers)

# # 1-41) 100개의 0을 가진 list 만들기 : 각 자리에 새로 계산한 값들을 저장해줄때.. 많이 씀 (근데 아래는 101개임)
#
# numbers = list()
# for _ in range(101):
#     numbers.append(0)
# print(numbers)

# 리스트 내포는 한줄임: [0 for _ in range(101)] >> 변수에 넣어줘야 함!

# # 1-42) for loop 으로 list의 원소 접근하기(2)
#
# scores = [10,20,30]
#
# # method1
# for score in scores:
#     print(score)
#
# # method2 - index로 접근하기!
# for score_idx in range(len(scores)): # 0,1,2
#     print(scores[score_idx])

# # 1-43) for loop으로 list의 원소 수정하기
#
# scores = [10,20,30,40,50]
#
# for score_idx in range(len(scores)):
#     scores[score_idx] += 10
#
# print(scores)

# # 1-44) 두 개의 list 접근하기
#
# list1 = [10,20,30]
# list2 = [100,200,300]
#
# for idx in range(len(list1)):
#     print(list1[idx], list2[idx])

# # 1-45) 수학점수들의 평균 구하기(3)
#
# scores = [10,20,30]
#
# # method1
# score_sum = 0
# n_student = 0
# for score in scores:
#     score_sum += score
#     n_student += 1
# score_mean = score_sum / n_student
# print('score mean: ', score_mean)
#
# # method2
# score_sum = 0
# for score_idx in range(len(scores)):
#     score_sum += scores[score_idx]
# score_mean = score_sum / len(scores)
# print('score mean: ', score_mean)

# # 1-46) Mean Subtraction(3) : 각각의 데이에서 평균을 빼고 그거의 평균을 다시 구하면 0 이 됨. >> 얘도 data scale기법
#
# scores = [10,20,30]
# score_mean = (scores[0] + scores[1] + scores[2]) / len(scores)
#
# # method1
# scores_ms = list()
# for score in scores:
#     scores_ms.append(score - score_mean)
# print(scores_ms)
#
# # method2
# for score_idx in range(len(scores)):
#     scores[score_idx] -= score_mean
# print(scores)

# # 1-47) 두 과목의 평균 구하기
#
# math_scores = [40,60,80]
# english_scores = [30,40,50]
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
#     score_sums[0] += math_scores[student_idx]
#     score_sums[1] += english_scores[student_idx]
# print('sums of scores: ', score_sums)
#
# for class_idx in range(n_class):
#     class_mean = score_sums[class_idx] / n_student
#     score_means.append(class_mean)
# print('means of scores: ', score_means)


# 1-48)
math_scores = [40,60,80]
english_scores = [30,40,50]

n_class = 2
n_student = len(math_scores)

score_sums = list()
score_means = list()

for _ in range(n_class):
    score_sums.append(0)

for student_idx in range(n_student):
    score_sums[0] += math_scores[student_idx]
    score_sums[1] += english_scores[student_idx]


for class_idx in range(n_class):
    class_mean = score_sums[class_idx] / n_student
    score_means.append(class_mean)

for student_idx in range(n_student):
    math_scores[student_idx] -= score_means[0]
    english_scores[student_idx] -= score_means[1]

print('Math scores after mean subtraction: ', math_scores)
print('English scores after mean subtraction: ', english_scores)



math_scores = [40,60,80]
english_scores = [30,40,50]

n_class = 2
n_student = len(math_scores)

score_sums = list()
score_means = list()

for _ in range(n_class):
    score_sums.append(0)

for student_idx in range(n_student):
    score_sums[0] += math_scores[student_idx]
    score_sums[1] += english_scores[student_idx]


for class_idx in range(n_class):
    class_mean = score_sums[class_idx] / n_student
    score_means.append(class_mean)

for student_idx in range(n_student):
    math_scores[student_idx] -= score_means[0]
    english_scores[student_idx] -= score_means[1]

print('Math scores after mean subtraction: ', math_scores)
print('English scores after mean subtraction: ', english_scores)



math_scores = [40,60,80]
english_scores = [30,40,50]

n_class = 2
n_student = len(math_scores)

score_sums = list()
score_means = list()

for _ in range(n_class):
    score_sums.append(0)

for student_idx in range(n_student):
    score_sums[0] += math_scores[student_idx]
    score_sums[1] += english_scores[student_idx]


for class_idx in range(n_class):
    class_mean = score_sums[class_idx] / n_student
    score_means.append(class_mean)

for student_idx in range(n_student):
    math_scores[student_idx] -= score_means[0]
    english_scores[student_idx] -= score_means[1]

print('Math scores after mean subtraction: ', math_scores)
print('English scores after mean subtraction: ', english_scores)













