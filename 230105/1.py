import matplotlib.pyplot as plt
import numpy as np

# np.random.seed(0)
#
# y_data = np.random.normal(loc=0, scale=1, size=(300,)) # loc 평균, scale 표준편차, size 몇개 데이터 뽑겠다 >> 이렇게 파라미터를 명시해주면 순서 안지켜도 됨. but 숫자만 넣으면 순서 지켜야 함!
#
# fig, ax = plt.subplots(figsize=(10,5))
# ax.plot(y_data)
#
# fig.tight_layout(pad=3) # pad 여백
# ax.tick_params(labelsize=25) # tick 글자 크기
#
# plt.show()

# -----------------------------------------------------------------------------------------

# # xtick의 step 바꾸기
#
#
# np.random.seed(0)
#
# y_data = np.random.normal(0, 1, (300,))
#
# fig, ax = plt.subplots(figsize=(10,5))
# ax.plot(y_data)
#
# fig.tight_layout(pad=3)
# ax.tick_params(labelsize=25)
#
# # # 1
# # x_ticks = np.arange(301, step=50)
# # ax.set_xticks(x_ticks)
#
# # 2
# x_ticks = np.arange(301, step=100)
# ax.set_xticks(x_ticks)
#
# plt.show()

# --------------------------------------------------------------------------------------------

# # np.arange > numpy에서 사용하는 range함수라고 생각하면 됨 / np.arange(301)하면 0~300까지 숫자 들어간 array 생성 / np.arange(시작점, 끝점, step)
# # https://jimmy-ai.tistory.com/45 (참고)
#
# np.random.seed(0)
#
# n_data = 100
# s_idx = 30
# x_data = np.arange(s_idx, s_idx + n_data)
# y_data = np.random.normal(0, 1, (n_data,))
#
# fig, ax = plt.subplots(figsize=(10,5))
# ax.plot(x_data, y_data)
#
# fig.tight_layout(pad=3)
# x_ticks = np.arange(s_idx, s_idx+n_data+1, 20) # 30~130까지 20 step
# ax.set_xticks(x_ticks)
#
# ax.tick_params(labelsize=25)
# ax.grid() # 그리드(점선)
#
# plt.show()

# ---------------------------------------------------------------------------------------------------

# # fig.subplots_adjust > fig.tight_layout과 비슷! 좀 더 자세히 보고 싶을때 앞에걸 사용(tight는 상하좌우 한번에, subplots_adjust는 따로따로 가능)
# # tick과 grid 차이 > 정확한 값을 확인하고 싶을때!
#
# np.random.seed(0)
#
# x_data = np.array([10,25,31,40,55,80,100])
# y_data = np.random.normal(0,1,(7,))
#
# fig, ax = plt.subplots(figsize=(10,5))
# ax.plot(x_data, y_data)
#
# fig.subplots_adjust(left=0.2)
# ax.tick_params(labelsize=25)
#
# # tick, grid
# ax.set_xticks(x_data) # x_data 찍어줌
# ylim = ax.get_ylim() # get_ylim > y축 범위 가져옴(그러니까 (최소,최대))
# # print(ylim) # (최소, 최대)
# yticks = np.linspace(ylim[0], ylim[1], 8) # 최소, 최대
# ax.set_yticks(yticks)
#
# plt.show()

# --------------------------------------------------------------------------------------------

# np.random.seed(0)
#
# x_data = np.random.normal(0, 1, (10,))
# y_data = np.random.normal(0, 1, (10,))
#
# fig, ax = plt.subplots(figsize=(10,10))
# ax.plot(x_data, y_data)
#
# plt.show()

# ---------------------------------------------------------------------------------------------

# 여러 그래프 그리기! / 평균이 다름! 그래서 그래프 다름

# n_data = 100
#
# random_noise1 = np.random.normal(0,1,(n_data,))
# random_noise2 = np.random.normal(1,1,(n_data,))
# random_noise3 = np.random.normal(2,1,(n_data,))
#
# # print(random_noise1)
#
# fig, ax = plt.subplots(figsize=(10,7))
#
# # 이게 x,y값이 다 나오는건가..???  y값들을 다 선으로 이어주는건가? y가 100개니까 x가 0~100까지 범위가 자동으로 잡히는듯?
# ax.plot(random_noise1)
# ax.plot(random_noise2)
# ax.plot(random_noise3)
#
# ax.tick_params(labelsize=20)
#
# plt.show()


# -----------------------------------------------------------------------------------------------

# # 데이터 개수 달라짐.. 잘 모르겠음..
#
# n_data1, n_data2, n_data3 = 200, 50, 10
#
# # linspace를 0~200에서 n_data만큼 균등하게 쪼갠거니까 암튼 x축의 범위는 0~200!!
# x_data1 = np.linspace(0, 200, n_data1)
# x_data2 = np.linspace(0, 200, n_data2)
# x_data3 = np.linspace(0, 200, n_data3)
#
# random_noise1 = np.random.normal(0, 1, (n_data1,))
# random_noise2 = np.random.normal(1, 1, (n_data2,))
# random_noise3 = np. random.normal(2, 1, (n_data3,))
#
# fig, ax = plt.subplots(figsize=(10,7))
#
# ax.plot(x_data1, random_noise1)
# ax.plot(x_data2, random_noise2)
# ax.plot(x_data3, random_noise3)
#
# ax.tick_params(labelsize=20)
#
# plt.show()

# ---------------------------------------------------------------------------------

# # tick에 label주는거
# # f스트링하는거랑 비슷.. 근데 r이 들어감 > r'&~~&' 달라 사이에 latex(레이텍)코드를 넣어줌(latex코드는 \로 보통 시작!)
#
# PI = np.pi
# t = np.linspace(-4*PI, 4*PI, 300)
# sin = np.sin(t)
# linear = 0.1*t # y=0.1x
#
# fig, ax = plt.subplots(figsize=(14, 7))
# ax.plot(t, sin)
# ax.plot(t, linear)
#
# ax.set_ylim([-1.5, 1.5])
#
# x_ticks = np.arange(-4*PI, 4*PI+0.1, PI)
# x_ticklabels = [str(i) + r'$\pi$'
#                 for i in range(-4, 5)]
#
# ax.set_xticks(x_ticks)
# ax.set_xticklabels(x_ticklabels)
#
# ax.tick_params(labelsize=20)
# ax.grid()
#
# plt.show()

# --------------------------------------------------------------------------------------------

# 사진
# 하나의 x로 여러 그래프를 그림
# subplots 부분이 바뀜(3,1,,,,) >> 3행1열 / 하나의 행렬로 보면 됨(그래프 하나가 행렬의 하나의 원소!)
# tan 그래프..끊어져야 하는데 다 연결되어 있음..(사실 오른쪽도 직선이 없어야 함) > matplotlib에서 lineplot 그릴때 오류..? 발산하는 그래프 그릴때 이런 오류가 발생함(왜냐하면 lineplot은 모든 데이터를 직선으로 이어주니까! 이런 문제 발생)
# 코드에 'y[:1][np.diff(y) < 0] = np.nan' 이런 코드를 추가해주면 tan그래프 잘 그려짐(np.diff 차이.. 나중 값에서 처음 값을 뺀게 np.diff(y)인데 얘가 음수인 부분을 받아서.. nan이 not a number라는 뜻인데.. 이 부분을 아예 숫자로 취급x (불리언 인덱싱 활용! T=1,F=0 >> 얘 어디서 쓰임?)
# axes[0] 처럼 인덱싱으로 접근함!

# PI = np.pi
# t = np.linspace(-4*PI, 4*PI, 1000)
# sin = np.sin(t)
# cos = np.cos(t)
# tan = np.tan(t)
#
# fig, axes = plt.subplots(3, 1, figsize=(7,10)) # 3행1열?!
#
# axes[0].plot(t, sin)
# axes[1].plot(t, cos)
# axes[2].plot(t, tan)
# axes[2].set_ylim([-5, 5]) # 유,무 차이점 보기(범위 설정해주면 좀 더 tan그래프)
#
#
# fig.tight_layout()
#
# plt.show()

# -------------------------------------------------------------------------------------------------

# for문: 각각의 axes(엑스)에 접근해서 각각의 그래프를 그려주는 것!  / 각각의 axes에 그려지는 데이터도 다름!
# t = np.linspace(-4*PI, 4*PI, 1000) > 여기까지하면 (1000,) 원소 1000개를 가진 벡터가 만들어짐
# reshape하면 재배열해주는거!!
# ex. 엄청 큰 데이터가 있음.. 5000 을 > 4개의 행으로 만들어주고 싶음. (열을 관심x) > reshape(4,-1) 하면 열 알아서 세팅됨.
# 벡터는 1차 tensor임!!(괄호가 하나임) / reshape하면 괄호가 2개 생김. (numpy array에서는 [] 이 괄호가 차원을 말한다고 생각하면 되는데.. [] 이거는 벡터고, [[]]는 행렬임. 완전 다름..흠)

# vstack: 데이터를 위아래로 쌓아줌. / hstack(horizontal stack)
# sin, cos, tan 데이터를 vstack으로 쌓아줌 >> [[].[],[]] 이런 행렬이 됨

# sharex... 위와 다른거.. x축이 하나임.. share x(x축을 공유하겠다!)

# 사진
# flatten : ..지금 axes를 여러개를 그리면 행렬로 인식함. 얘를 벡터로 만들어 주는 것! flat이 필요할 때는.. 3행2열짜리 flatten을 해줘야 함.. for문 돌리려면 차례대로 접근하게 해줘야 하는데.. 그럼 flatten해줘야 함. flatten하면 1행 6열로 됨)
# flat, flatten 다 벡터로 만들어주는 것. flat을 하면 인덱싱이 바로 가능해짐 ex. axes.flat[0] / 1차 array에 인덱싱으로 바로 접근 가능


######## reshape&flatten, axes.flat, vstack > hstack 여러가지 실험하면서 결과 나오는거 보면서..뜯어가면서 하기!!


PI = np.pi
t = np.linspace(-4*PI, 4*PI, 1000).reshape(1,-1) # 이미 flat해준거 아님? / [] >> [[]] (reshape하니까) / flat해서 어차피 벡터로 만들어야 하면.. 그냥 reshape안해도 되는거 아닌가? x값에 넣으려면 행렬이어야 해서 그런가?! >>>reshape하면 밑에서 flatten안해줘도 됨! / reshape안하면 밑에서 해줘야 함.
sin = np.sin(t) # 행렬 값을 넣으면 행렬이 나옴! 행렬 넣든 벡터 넣든 상관x
cos = np.cos(t)
tan = np.tan(t)
data = np.vstack((sin,cos,tan)) # y값..들..
# print(t) # 행렬..? np. 한번에 넣으려면 행렬이어야 되는듯.?
# print(t.flatten()) # 벡터..? for문 돌려서 하나하나 넣어주려면 벡터여야 하는듯?
# print(data[1]) # cos(t)값들
# print(data) # [[sin(t)값들], [cos(t)값들], [tan(t)값들]]

title_list = [r'$\sin(t)$', r'$\cos(t)$', r'$\tan(t)$']
x_ticks = np.arange(-4*PI, 4*PI+PI, PI)
x_ticklabels = [str(i) + r'$\pi$' for i in range(-4, 5)]

fig, axes = plt.subplots(3, 1,             # 행렬형태.. 3by1모양으로 그리는것! / 여러개 그래프니까 axes임
                         figsize=(7,10),
                         sharex=True)

for ax_idx, ax in enumerate(axes.flat):   # 그래프3개를 쭉 펴줌 >> .flat없어도 결과 값음! 어차피 1열인 상태면 어차피 차례대로 접근. 하지만 column개수가 늘어나면 flat해서 접근해줘야 함!(접근 순서를 명확하게 정해줘야 하니까!) 사실 그래서 여기서는 필요x(1열이니까) >>> 얘는 axes 3개인 상태(3행1열)
    ax.plot(t.flatten(), data[ax_idx])    # 그래프 만들때 식에 x값 넣으려면 벡터 형태여야 함?(왜 flatten..) / 위에 처럼 np.식에 값 한꺼번에 넣으려면 행렬이어야 되는건가?(reshape) / 벡터는 그래서 for문?! /// 그래서 사실 위에서 reshape(벡터>행렬) 안해줬으면 flatten할 필요도 x. reshape 안했으면 그냥 벡터 형태니까.
    ax.set_title(title_list[ax_idx],
                 fontsize=30)
    ax.tick_params(labelsize=20)
    ax.grid()
    if ax_idx == 2:
        ax.set_ylim([-3, 3])

fig.subplots_adjust(left=0.1, right=0.95,
                    bottom=0.05, top=0.95)
axes[-1].set_xticks(x_ticks)
axes[-1].set_xticklabels(x_ticklabels)

plt.show()

# ------------------------------------------------------------------------------------

# axvline > vertical line 그려주는 것
# axhline > horizontal line 그려주는 것
# 오른쪽 그림같이는 원래 잘 안함. (비율로 함. 코드보면! y축에서 0.2ㅇ에서 0.8까지 vline그려줌)(그냥 보여주려고)
# hline, vline 많이 씀 > 점근선, min, max, 기준선 등에 많이 활용됨!

# # vline
# fig, ax = plt.subplots(figsize=(7,7))
#
# ax.set_xlim([-5,5])
# ax.set_ylim([-5,5])
#
# ax.axvline(x=1,
#            color='black',
#            linewidth=1)
#
# plt.show()
#
# # 비율로 vline 그리기
# fig, ax = plt.subplots(figsize=(7,7))
#
# ax.set_xlim([-5,5])
# ax.set_ylim([-5,5])
#
# ax.axvline(x=1,
#            ymax=0.8, ymin=0.2,
#            color='black',
#            linewidth=1)
#
# plt.show()

# --------------------------------------------------------------------------------------

# # hline
# fig, ax = plt.subplots(figsize=(7,7))
#
# ax.set_xlim([-5, 5])
# ax.set_ylim([-5, 5])
#
# ax.axhline(y=1,
#            color='black',
#            linewidth=1)
#
# plt.show()
#
# # 비율로 hline 그리기
# fig, ax = plt.subplots(figsize=(7,7))
#
# ax.set_xlim([-5, 5])
# ax.set_ylim([-5, 5])
#
# ax.axhline(y=1,
#            xmax=0.8, xmin=0.2,
#            color='black',
#            linewidth=1)
#
# plt.show()

# -------------------------------------------------------------------------------

# # ls = linestyle, lw = linewidth
#
# x = np.linspace(-4*np.pi, 4*np.pi, 200)
# sin = np.sin(x)
#
# fig, ax = plt.subplots(figsize=(10,5))
# ax.plot(x, sin)
# ax.axhline(y=1, ls=':', lw=1, color='gray')
# ax.axhline(y=-1, ls=':', lw=1, color='gray')
#
# plt.show()

# ---------------------------------------------------------------------------------

# 다양한 라인스타일, 마커스타일을 사용하고 알아야 하는 이유는? 단순히 이뻐서x / 구분하려고!(논문이나 책에서 인쇄할때 color인경우 거의x, 대체로 흑백! 데이터별로 구분해주려고 마커나 라인스타일로 구분해줌!)

# line styles
# 그래프 색 따로 지정해주지 않아도 디폴트 컬러맵인 tab10으로 다 다른색으로 나옴!

# x_data = np.array([0,1])
# y_data = x_data
#
# fig, ax = plt.subplots(figsize=(10,10))
#
# ax.plot(x_data, y_data)
#
# ax.plot(x_data, y_data+1,
#         linestyle='dotted')
#
# ax.plot(x_data, y_data+2,
#         ls='dashed')
#
# ax.plot(x_data, y_data+3,
#         ls='dashdot')
#
# plt.show()

# ------------------------------------------------------------------------------------

# # 주로 이 방법을 많이 씀!! 잘 익혀두자!
#
# x_data = np.array([0,1])
# y_data = x_data
#
# fig, ax = plt.subplots(figsize=(10,10))
#
# ax.plot(x_data, y_data)
#
# ax.plot(x_data, y_data+1,
#         linestyle=':')
#
# ax.plot(x_data, y_data+2,
#         ls='--')
#
# ax.plot(x_data, y_data+3,
#         ls='-.')
#
# plt.show()

# -------------------------------------------------------------------------------

# PI = np.pi
# t = np.linspace(-4*PI, 4*PI, 300)
# sin = np.sin(t)
#
# fig, ax = plt.subplots(figsize=(10,7))
#
# ax.plot(t, sin)
#
# ax.axhline(y=1,
#            linestyle=':')
# ax.axhline(y=-1,
#            linestyle=':')
# plt.show()

# ---------------------------------------------------------------------------------

# PI = np.pi
# t = np.linspace(-4*PI, 4*PI, 300)
# sin = np.sin(t)
#
# fig, ax = plt.subplots(figsize=(10,7))
#
# ax.plot(t, sin,
#         color='black')
# ax.axhline(y=1,
#            linestyle=':',
#            color='red')
# ax.axhline(y=-1,
#            linestyle=':',
#            color='blue')
#
# plt.show()

# ------------------------------------------------------------------------------------

# PI = np.pi
# t = np.linspace(-4*PI, 4*PI, 50) # 300으로 하니까 너무 징그러월..
# sin = np.sin(t)
#
# fig, ax = plt.subplots(figsize=(10,7))
#
# ax.plot(t, sin,
#         color='black')
# ax.plot(t, sin+1,
#         marker='o',       # 이렇게 'marker='와 같이 파라미터를 지정해서 마커를 넣어주면 lineplot+scatterplot / 'o'만 입력하면 scatterplot
#         color='black')
# ax.plot(t, sin+2,
#         marker='D',
#         color='black')
# ax.plot(t, sin+3,
#         marker='s',
#         color='black')
#
# plt.show()

# ------------------------------------------------------------------

# # 하트 그리기
# from pylab import *
# x = linspace(-1.6, 1.6, 10000)
# f = lambda x: (sqrt(cos(x)) * cos(200 * x) + sqrt(abs(x)) - 0.7) * \
#     pow((4 - x * x), 0.01)
# plot(x, list(map(f, x)))
# show()

# ------------------------------------------------------------------------

# # 마커사이즈
#
# PI = np.pi
# t = np.linspace(-4*PI, 4*PI, 50)
# sin = np.sin(t)
#
# fig, ax = plt.subplots(figsize=(10, 7))
#
# ax.plot(t, sin+1,
#         marker='o',
#         color='black',
#         markersize='15')
#
# plt.show()

# -------------------------------------------------------------------------

# # facecolor, edgecolor
#
# PI = np.pi
# t = np.linspace(-4*PI, 4*PI, 50)
# sin = np.sin(t)
#
# fig, ax = plt.subplots(figsize=(10, 7))
#
# ax.plot(t, sin+1,
#         marker='o',
#         color='black',
#         markersize='15',
#         markerfacecolor='r',
#         markeredgecolor='b')
#
# plt.show()

# ----------------------------------------------------------------------------

# # markeredgewidth
#
# PI = np.pi
# t = np.linspace(-4*PI, 4*PI, 50)
# sin = np.sin(t)
#
# fig, ax = plt.subplots(figsize=(10, 7))
#
# ax.plot(t, sin+1,
#         marker='o',
#         color='black',
#         markersize='15',
#         markerfacecolor='r',
#         markeredgecolor='b',
#         markeredgewidth=3)
#
# plt.show()

# --------------------------------------------------------

# # :ob 순서상관x / 한꺼번에 씀(linestyle, marker, color)
#
# x_data = np.array([1,2,3,4,5])
#
# fig, ax = plt.subplots(figsize=(10,10))
#
# ax.plot(x_data,
#         linestyle=':',
#         marker='o',
#         color='b')
#
# plt.show()

# # 한번에 모아쓰기
#
# y_data = np.array([1,2,3,4,5])
#
# fig, ax = plt.subplots(figsize=(10,10))
#
# ax.plot(y_data, ':ob') # ':bo', 'bo:' 순서 상관x / plot에 값 하나만 넣으면 y값으로 인식! x값은 알아서 됨! 기억기억!(y값이니까 y_data라고 이름 짓는게 good!)
#
# plt.show()

# -------------------------------------------------------------------------------

# # 데이터를 class별로 label해준 것을 legend라고 함!
# # legend를 그려줄때는 plot을 그릴때마다 어떤 label을 가지고 있는지 적어줘야(labeling) legend를 호출했을 때 legend가 잘 생성됨!
#
# np.random.seed(0)
#
# n_data = 100
# random_noise1 = np.random.normal(0,1,(n_data,))
# random_noise2 = np.random.normal(1,1,(n_data,))
# random_noise3 = np.random.normal(2,1,(n_data,))
#
# fig, ax = plt.subplots(figsize=(10, 7))
# ax.tick_params(labelsize=20)
#
# ax.plot(random_noise1,
#         label='random noise1') # 이렇게 plot 그릴때마다 labeling을 꼭 해줘야 한다!!
# ax.plot(random_noise2,
#         label='random noise2')
# ax.plot(random_noise3,
#         label='random noise3')
#
# ax.legend()
# # ax.legend(fontsize=20) # fontsize뿐만 아니라 위치도 달라짐! > legend location이 자동적으로 최적의 위치를 찾아서 거기 들어감(따로 설정안하면 디폴트값인 best, 0 으로 됨)
#
# plt.show()

# ------------------------------------------------------------------------------

# # # legend locations
# #
# n_data = 100
# random_noise1 = np.random.normal(0, 1, (n_data,))
# random_noise2 = np.random.normal(1, 1, (n_data,))
# random_noise3 = np.random.normal(2, 1, (n_data,))
#
# fig, ax = plt.subplots(figsize=(10, 10))
#
# ax.plot(random_noise1,
#        label='random noise1')
# ax.plot(random_noise2,
#         label='random noise2')
# ax.plot(random_noise3,
#         label='random noise3')
#
# ax.legend(fontsize=20,
#           loc='upper right') # 위치 여러개 다 해보기
#
# plt.show()

# ---------------------------------------------------------------------------












