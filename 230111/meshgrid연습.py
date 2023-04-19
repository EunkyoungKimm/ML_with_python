import numpy as np
import matplotlib.pyplot as plt

a = np.linspace(1,100)
b = np.linspace(1,100)
x,y = np.meshgrid(a, b) # a,b를 meshgrid인자로 넘겨줌 > 인자값 첫 두개는 결국 그리드를 찍을 x값 범위 배열, y값 범위 배열 의 의미

print(x, y)

plt.scatter(x,y)

plt.show()
