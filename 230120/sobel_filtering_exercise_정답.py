import numpy as np
import matplotlib.pyplot as plt

def get_check_pattern_img():
    white_patch = 1 * np.ones(shape=(10, 10))
    black_patch = 0 * np.ones(shape=(10, 10))

    img1 = np.hstack([white_patch, black_patch])
    img2 = np.hstack([black_patch, white_patch])
    img = np.vstack([img1, img2])

    img = np.tile(img, reps=[2, 2])
    return img

img = get_check_pattern_img()

sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])

sobel_y = np.array([[1, 2, 1],
                    [0, 0, 0],
                    [-1, -2, -1]])

H, W = img.shape  # 테스트 이미지의 H, W 할당하기 => H_, W_ 계산하기 위해서
F = sobel_x.shape[0]  # filter size 계산하기 => 정사각형 모양이니까 filter의 높이를 F로

H_, W_ = H - F + 1, W - F + 1  # 출력 이미지의 H_, W_ 공식으로 계산
img_filtered = np.zeros(shape=(H_, W_))  # filtering된 값을 저장할 image 초기화하기

for h_idx in range(H_):  # 아래쪽으로 이동하는 것은 오른쪽으로 이동한 것이 끝난 뒤에
    for w_idx in range(W_):  # 오른쪽으로 이동하는 것 먼저 진행하기
        window = img[h_idx: h_idx + F,
                 w_idx: w_idx + F]  # (3, 3) window extraction => 추출!

        ''' ***** 내가 가지고 있는 filter와 window가 비슷한 특징을 가진 window인지 연산 *****'''
        z = np.sum(window * sobel_x)
        ''' ***** 내가 가지고 있는 filter와 window가 비슷한 특징을 가진 window인지 연산 *****'''
        img_filtered[h_idx, w_idx] = z

fig, axes = plt.subplots(1, 2, figsize=(12, 8))
axes[0].imshow(img, cmap='gray')
axes[1].imshow(img_filtered, cmap='gray')

for ax in axes:
    ax.tick_params(left=False, labelleft=False,
                   bottom=False, labelbottom=False)
plt.show()