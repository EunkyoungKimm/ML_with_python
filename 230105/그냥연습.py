# https://nittaku.tistory.com/117

import matplotlib.pyplot as plt
import matplotlib

fig = plt.figure() # 내부가 텅빈 figure 생성 > 빈 도화지

'''
1) figure -> subplot 으로 세분화해서 그릴 준비하기
 - fig에 subplot 추가하기
 - add_subplot() 을 호출하여 좌표평면(axes)을 나타내는 axes 변수에 받아줌
 - add_subplot(몇행, 몇열, 몇번째 성분) 에 그릴지.. 앞에와 같은 '인자'를 가지고 있는 함수!
'''

ax1 = fig.add_subplot(2,2,1) #2x2 중 1번째 위치에 빈 좌표평면(ax1)이 그려짐 (1행 1열)
ax2 = fig.add_subplot(2,2,2) # 1행 2열
ax3 = fig.add_subplot(2,2,3) # 2행 1열

plt.show()

'''
2) numpy, pandas로 데이터 생성후, figure의 subplot마다 그려보기
- '''