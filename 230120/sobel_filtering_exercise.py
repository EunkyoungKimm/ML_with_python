# sobel filtering exercise

import numpy as np
import matplotlib.pyplot as plt

white_patch = 255*np.ones(shape=(10,10))
black_patch = 0*np.ones(shape=(10,10))

### check_pattern_image
img1 = np.hstack([white_patch, black_patch])
img2 = np.hstack([black_patch, white_patch])
img = np.vstack([img1, img2])

check_pattern_image = np.tile(img, reps=[2,2]) # (40,40)
# print(check_pattern_image)

### sobel filtering
# filter
x_filter = np.array([[-1,0,1],
                     [-2,0,2],
                     [-1,0,1]])

H, W = check_pattern_image.shape # 전체 이미지의 H,W
F = x_filter.shape[0]
H_ = H - F + 1
W_ = W - F + 1

filtered_data = np.zeros(shape=(H_, W_))
for h_idx in range(H_):
    for w_idx in range(W_):
        ### window
        window = check_pattern_image[h_idx : h_idx + F, w_idx : w_idx + F]
        z = np.sum(window * x_filter) # matmul은 언제..? 이게 matmul? => np.sum(np.matmul(window, x_filter)) 이렇게 하면 다 0 나옴.. ==> matmul 동작 원리, 결과물을 내가 잘 모르는 듯! 다시 공부해보기!

        filtered_data[h_idx, w_idx] = z
print(filtered_data)


fig, axes = plt.subplots(1, 2, figsize=(10,5))
axes[0].imshow(check_pattern_image, cmap='gray')
axes[1].imshow(filtered_data, cmap='gray')

# remove tick and tick label
axes[0].tick_params(left=False, labelleft=False,
               bottom=False, labelbottom=False)
axes[1].tick_params(left=False, labelleft=False,
               bottom=False, labelbottom=False)


fig.tight_layout()
plt.show()