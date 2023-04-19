import matplotlib.pyplot as plt
import numpy as np

# create a new figure 그래프 그릴 도화지 만드는 것
fig = plt.figure()
plt.show()

fig, ax = plt.subplots(figsize=(7,7))
plt.show()
# object = 데이터+기능 / fig, ax를 생성하면 좋은 이유는 얘네 저장한걸 나중에 불러서 쓸 수 있음(단순히 plt.plot, plt.scatter하면 이게 안됨. 나중에 재사용x)