# 코드에 for문을 없앨 수록 시간 복잡도 면에서 좋다.
# 나중에 코드가 길어지면 그 연산 속도 차이가 엄청 크다.

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

# numpy로 만드는 이유. for문 안써도 numpyp로 넣어주면 그냥 x,y쭉 들어가서 그런거같음! / df로 해주고, numpy로..
X = athletes_df[['SPEED', 'AGILITY']].to_numpy() # [[]] 여러개 하려면 이중으로 써야 하나?(여러개를 한 리스트 안에 넣어주려고?) / numpy형태로 바꾸는 이유?
y = athletes_df['DRAFT'].values # 값 뽑아낼때는 이렇게?
# print(X)
# print(y)

fig, ax = plt.subplots(figsize=(10, 10))

X_pos, X_neg = X[y == 'yes'], X[y == 'no'] # 이렇게도 나눌 수 있음!
# print(X_pos)
# print(X_pos[:, 0])
# print(X_neg)

ax.scatter(X_pos[:, 0], X_pos[:, 1], color='blue',
           marker='+', s=130, label='yes')
ax.scatter(X_neg[:, 0], X_neg[:, 1], color='red',
           marker='^', s=100, label='no')
ax.set_xlabel("Speed", fontsize=20)
ax.set_ylabel("AGILITY", fontsize=20)
ax.legend(fontsize=15)
ax.tick_params(labelsize=15)

plt.show()