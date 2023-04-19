# scatter에 hatch='/' / '//'///// > 2차원 평면이지만.. 차원을 좀 더 추가 해주기 위해 hatch 해줌 / hatch랑 색깔 이용해서 다 분류해주기 위해.
# 4개 feature를 다 표현해줌. 2차원 평면에다가(다양한 feature를 표현하기 위해 hatch, 색깔 등으로 다 표현해줌)
# colormap = tab20(17,17 > 해치가 다름) / 색이 다 다르면 너무 난잡
# fake scatter > legend찍어줄 수 있도록 (ax.scatter([], []) 빈리스트 줌

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# data
countries = ['Australia', 'Austria', 'Belgium', 'Canada', 'Chile',
             'Czech Republic', 'Denmark', 'Estonia', 'Finland', 'France',
             'Germany', 'Greece', 'Hungary', 'Iceland', 'Ireland',
             'Israel', 'Italy', 'Japan', 'Korea', 'Luxembourg',
             'Mexico', 'Netherlands', 'New Zealand', 'Norway', 'Poland',
             'Portuagl', 'Slovak Republic', 'Slovenia', 'Spain', 'Sweden',
             'Switzerland', 'Turkey', 'United Kingdom', 'United States']

population_denstiy = [3, 101, 367, 4, 23,
                      133, 130, 29, 16, 117,
                      227, 86, 106, 3, 65,
                      365, 203, 337, 501, 207,
                      60, 406, 17, 13, 122,
                      116, 110, 103, 91, 21,
                      194, 97, 257, 32]
private_expenditure = [1.3, 8.9, 10.0, 10.9, 12.8,
                       13.0, 13.2, 13.4, 13.5, 14.8,
                       15.0, 15.8, 16.7, 17.1, 17.2,
                       17.4, 18.1, 18.1, 18.7, 18.9,
                       19.0, 19.0, 19.5, 19.9, 20.0,
                       21.5, 21.6, 23.0, 24.0, 25.0,
                       25.9, 26.7, 28.0, 34.3]
gdp = [38.7, 37.4, 33.6, 37.5, 16.4,
       24.5, 33.2, 19.3, 32.1, 32.0,
       36.2, 19.8, 17.8, 37.7, 37.7,
       29.4, 26.6, 32.0, 31.0, 67.9,
       13.4, 38.4, 27.0, 48.2, 18.9,
       20.9, 21.8, 24.2, 26.8, 36.2,
       42.5, 13.9, 35.6, 45.7]



gdp = np.array(gdp)
n_country = len(countries)
x_data = population_denstiy
y_data = private_expenditure
cmap = cm.get_cmap('tab20')

colors = [cmap(i) for i in range(int(n_country/2))]

fig, ax1 = plt.subplots(figsize=(20, 10))
ax2 = ax1.twinx() # 하나의 좌표평면에 여러 그래프 그릴때!(이렇게 해야 legend 따로 추가 가능!)
ax3 = ax1.twinx() # legend 마커사이즈 조절 일정하게 하려고

ax1.set_xlabel('Population Density(lnh./km2)', fontsize=30)
ax1.set_ylabel('Private Expenditure', fontsize=30)

ax1.set_ylim([-2,40])

ax1.tick_params(labelsize=15)

ax2.tick_params(right=False,
                labelright=False)
ax3.tick_params(right=False,
                labelright=False)

# ax1.scatter
for country_idx, country in enumerate(countries):
    # print(country_idx, country)
    if country_idx < n_country/2:
        ax1.scatter(x_data[country_idx], y_data[country_idx],
                    s=gdp[country_idx]*70,
                    facecolor=colors[country_idx],
                    hatch='//',
                    alpha=0.5,
                    label=country)
    else:
        ax1.scatter(x_data[country_idx], y_data[country_idx],
                    s=gdp[country_idx]*70,
                    facecolor=colors[country_idx-(int(n_country/2))],
                    hatch='.',
                    alpha=0.5,
                    label=country)

# ax2.scatter >> fake scatter
s_size = [10, 25, 40, 55]
for i in range(len(s_size)):
    ax2.scatter([], [],
                s=s_size[i]*70,
                facecolor=colors[0],
                label=s_size[i])

# ax3.scatter
for country_idx, country in enumerate(countries):
    if country_idx < n_country/2:
        ax3.scatter([], [],
                    s=300,
                    facecolor=colors[country_idx],
                    hatch='//',
                    alpha=0.5,
                    label=country)
    else:
        ax3.scatter([], [],
                    s=300,
                    facecolor=colors[country_idx-(int(n_country/2))],
                    hatch='.',
                    alpha=0.5,
                    label=country)


ax2.legend(loc='lower center',
           bbox_to_anchor=(0.5,1.03),
           ncol=4,
           title='GDP Value',
           title_fontsize=30,
           fontsize=20,
           labelspacing=1,
           columnspacing=2,
           edgecolor='white') # 박스 테두리 지우기

ax3.legend(loc='center left',
           bbox_to_anchor=(1,0.5),
           fontsize=10,
           ncol=2,
           labelspacing=2,
           columnspacing=2)

ax1.grid()

fig.tight_layout()


plt.show()








# ax1.legend(loc='center left',
#            bbox_to_anchor=(1,0.5),
#            fontsize=10,
#            markerscale=0.4, # 마커 크기 줄였다..ㅎ(default: 1.0) / 마커크기 일정하게 하는거.. 뭘까..?!?!? >> fake scatter처럼 해보기!!!
#            ncol=2,
#            labelspacing=2,
#            columnspacing=2)




















