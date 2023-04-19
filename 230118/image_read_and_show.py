# edge detection: pixel값이 변하는 곳을 찾아냄. 왜함? edge에는 상당히 많은 정보들이 담겨있음. edge만 봐도 이 이미지가 어떤 이미지인지 파악 가능!
# 이미지 판독, 글씨 판독, object dectection할 때 가장 먼저 하는게 edge부터 파악! edge detection장점은 연산량이 매우 적음 > sobel filter!!!
# 딥러닝 네트워크 연산중에 cnn이 있는데, 얘 연산 원리가 sobel filter랑 완전 같음!

# Lenna
# playboy 잡지 모델 사진임!
# 이거 수업할때 거의 이 사진 사용! 완전 유명한 사진임! playboy 이미지학회에 소송까지 걸었는데 걍 계속 씀.(이 분 할머니 됐을 때 공로상까지 줌. 이미지 처리에 큰 기여를 해서!)

import matplotlib.pyplot as plt

# # ax.imread
#
# img = plt.imread('./Lenna.png') # 이미지 읽기, 가져오기
# print(img.shape)
#
# fig, ax = plt.subplots(figsize=(10,10))
# ax.imshow(img) # ax.imshow() 이미지를 보여주는 함수
#
# plt.show()

# --------------------------------------------------------------------------------------------
# image coordinates > 이만큼 croop한거!(잘라옴)
# (핸드폰에 사진o) 컬러이미지는 rgb 매트릭스 3개로, 3차 tensor임!(순서가 (H,W,길이-컬러rgb) >> 이미지사이즈&모든 컬러를 가져오겠다!)
# 픽셀 데이터가 스칼라가 아닌 벡터이므로 이미지 데이터는 (세로픽셀수x가로픽셀수) 형태의 2차원 배열로 표현하지 못하고, (세로픽셀수x가로픽셀수x색채널) 형태의 3차원 배열로 저장한다.

# img = plt.imread('./Lenna.png')
# print(img.shape)
#
# fig, axes = plt.subplots(1,2, figsize=(14,7))
# fig.tight_layout()
# axes[0].tick_params(labelsize=20)
# axes[1].tick_params(labelsize=20)
#
# axes[0].imshow(img)
#
# img_cropped = img[:300, :300, :]
# axes[1].imshow(img_cropped) # 순서가 (H,W,길이) >> 이미지사이즈(h,w) & 모든 컬러를 가져오겠다!
#
# plt.show()

# ----------------------------------------------------------------------------------------------
# 우리가 원하는 부분만 가져와서 slicing하는 것!

# img = plt.imread('./Lenna.png')
# print(img.shape)
#
# fig, axes = plt.subplots(1,2, figsize=(14,7))
#
# axes[0].tick_params(labelsize=20)
# axes[1].tick_params(labelsize=20)
#
# axes[0].imshow(img)
#
# img_cropped = img[200:400, 150:350, :]
# axes[1].imshow(img_cropped)
#
# plt.show()

# ---------------------------------------------------------------------------------------------
# r,g,b값을 뽑았는데 초록색처럼 보이는 이유는?
# 행렬은 흑백 이미지다.. 비리디스라스 디폴트 컬러맵이 적용되서 이렇게 보이는거임!
# 나중에 그래이 색을 입혀주면 흑백으로 바뀜!

# img = plt.imread('./Lenna.png')
# print(img.shape)
#
# r_img = img[:,:,0]
# g_img = img[:,:,1]
# b_img = img[:,:,2]
#
# print(r_img.shape)
# print(g_img.shape)
# print(b_img.shape)
#
# fig, axes = plt.subplots(2, 2, figsize=(15,15))
# for ax in axes.flatten():
#     ax.tick_params(labelleft=False, labelbottom=False) # tick은 있음. tick의 label만 없어짐!
#
#     for spine_loc, spine in ax.spines.items(): # spine > 그래프 겉 테두리
#         # print(spine_loc, spine) # spine_loc > left, right, bottom, top / spine > Spine, Spine, Spine, Spine
#         spine.set_visible(False)
#
# fig.tight_layout()
#
# axes[0, 0].imshow(img)
# axes[0, 1].imshow(r_img)
# axes[1, 0].imshow(g_img)
# axes[1, 1].imshow(b_img)
#
# plt.show()

# -------------------------------------------------------------------------------------------
# 이미지는 8비트임. 총 256가지 표현 가능! (0~255) > 픽셀 하나하나 마다 다 다른 값을 가지고 있음(그래서 명암 차이 이런것도 다 보이는 거임)
# 00000000 > 검정색 / 11111111 > 흰색

# img = plt.imread('./Lenna.png')
# print(img.shape)
#
# r_img = img[:,:,0]
# g_img = img[:,:,1]
# b_img = img[:,:,2]
#
# fig, axes = plt.subplots(2, 2, figsize=(15,15))
# for ax in axes.flatten():
#     ax.tick_params(labelleft=False, labelbottom=False)
#     for spine_loc, spine in ax.spines.items():
#         spine.set_visible(False)
#
# fig.tight_layout()
#
# axes[0, 0].imshow(img)
# axes[0, 1].imshow(r_img, cmap='gray')
# axes[1, 0].imshow(g_img, cmap='gray')
# axes[1, 1].imshow(b_img, cmap='gray')
#
# plt.show()

# -------------------------------------------------------------------------------------
# colormap에는 continuous한거랑 descriptive?한 map이 있음!

# img = plt.imread('./Lenna.png')
#
# r_img = img[:,:,0]
# g_img = img[:,:,1]
# b_img = img[:,:,2]
#
# fig, axes = plt.subplots(2,2, figsize=(15,15))
# for ax in axes.flatten():
#     ax.tick_params(labelleft=False, labelbottom=False)
#     for spine_loc, spine in ax.spines.items():
#         spine.set_visible(False)
# fig.tight_layout()
#
# axes[0,0].imshow(img)
# axes[0,1].imshow(r_img, cmap='Reds_r')
# axes[1,0].imshow(g_img, cmap='Greens_r')
# axes[1,1].imshow(b_img, cmap='Blues_r')
#
# plt.show()

# ------------------------------------------------------------------------

# img = plt.imread('./rgb.png')
#
# r_img = img[:,:,0]
# g_img = img[:,:,1]
# b_img = img[:,:,2]
#
# fig, axes = plt.subplots(2,2, figsize=(15,15))
# for ax in axes.flatten():
#     ax.xaxis.set_visible(False) # 이거 대신 ax.tick_params(labelleft=False, labelbottom=False, left=False, bottom=False)
#     ax.yaxis.set_visible(False)
#     for spine_loc, spine in ax.spines.items():
#         spine.set_visible(False)
# fig.tight_layout()
#
# axes[0,0].imshow(img)
# axes[0,1].imshow(r_img, cmap='Reds') # Reds_r > 이 colormap 가져와서 반대로 먹여준다는 것! / bwr > 하면 yes가 빨간색, no가 파란색 되서 bwr_r한거임!
# axes[1,0].imshow(g_img, cmap='Greens')
# axes[1,1].imshow(b_img, cmap='Blues')
#
# plt.show()
# ----------------------------
# img = plt.imread('./rgb.png')
#
# r_img = img[:,:,0]
# g_img = img[:,:,1]
# b_img = img[:,:,2]
#
# fig, axes = plt.subplots(2,2, figsize=(15,15))
# for ax in axes.flatten():
#     ax.tick_params(labelleft=False, labelbottom=False, left=False, bottom=False) # labelleft,labelbottom=False 대신 labelsize=0해주면 됨!
#     ax.yaxis.set_visible(False)
#     for spine_loc, spine in ax.spines.items():
#         spine.set_visible(False)
# fig.tight_layout()
#
# axes[0,0].imshow(img)
# axes[0,1].imshow(r_img, cmap='Reds') # Reds_r > 이 colormap 가져와서 반대로 먹여준다는 것! / bwr > 하면 yes가 빨간색, no가 파란색 되서 bwr_r한거임!
# axes[1,0].imshow(g_img, cmap='Greens')
# axes[1,1].imshow(b_img, cmap='Blues')
#
# plt.show()

# ----------------------------------------------------------------------------------------------------------------------

## sobel filtering!!!!
# 개념: (핸드폰 사진)    filter들임!!!!(검,흰....) 얘네들을 이미지에다가 전부다 돌리는거!! 이동하면서 이미지를 판독. / 작은값을 반환해줌(이 필터랑 비슷한게 아닌거를!) > 이 필터들을 이용해서 edge를 가져옴!

import numpy as np
import matplotlib.pyplot as plt
# ----------------------------------------------------------------------------------------------------------
# # np.ones: 입력된 shape의 array를 만들고, 모두 1로 채워주는 함수
# tmp = np.ones(shape=(2, 3)) # 2by3차리 행렬을 만들어줌
# print(tmp, '\n')
#
# # ndarray에 scalar를 곱하면 원소별 곱셈
# tmp2 = 10*tmp # tmp에 들어있는 모든 원소에 10을 곱해라
# print(tmp2)

# ------------------------------------------------------------------------------------------------------------
# check pattern image
# white_patch = 255*np.ones(shape=(10,10)) # (10,10)짜리 흰색 패치 만들기 => 255로 채우기
# black_patch = 0*np.ones(shape=(10,10)) # (10,10)짜리 검은색 패치 만들기 => 0으로 채우기
#
# img1 = np.hstack([white_patch, black_patch]) # [흰,검]의 (10,20) 이미지 만들기
# img2 = np.hstack([black_patch, white_patch]) # [검,흰]의 (10,20) 이미지 만들기
# img = np.vstack([img1, img2]) # img1(1행), img2(2행) vstack으로 쌓아줌
#
# fig, ax = plt.subplots(figsize=(8,8))
# ax.imshow(img, cmap='gray') # 이미지 띄우기, 흑백이미지를 만들 것이므로 colormap은 'gray'로
#
# ax.tick_params(left=False, labelleft=False,
#                bottom=False, labelbottom=False)
#
# plt.show()

# ----------------------------------------------------------------------------------

# white_patch = 255*np.ones(shape=(10,10))
# black_patch = 0*np.ones(shape=(10,10))
#
# img1 = np.hstack([white_patch, black_patch, white_patch])
# img2 = np.hstack([black_patch, white_patch, black_patch])
# img = np.vstack([img1, img2, img1])
#
# fig, ax = plt.subplots(figsize=(8,8))
# ax.imshow(img, cmap='gray')
#
# ax.tick_params(left=False, labelleft=False,
#                bottom=False, labelbottom=False)
#
# plt.show()

# ---------------------------------------------------------------------------------
# np.repeat, np.tile > 이 함수를 이용해 복잡한 체크패턴 이미지 만들 수 있음

# data = np.arange(5) # [0 1 2 3 4] 1차원 array
#
# # np.repeat => 원소별 반복 (반복할 데이터, 횟수)
# print('repeat:', np.repeat(data, repeats=3))
#
# # np.tile => 전체 패턴 반복
# print('repeat:', np.tile(data, reps=3))

# ----------------------------------------------------------------------------------
# # np.repeat >원소별
# # axis..어떤 방향으로
# data = np.arange(6).reshape(2, 3)
# print(data)
#
# print('repeat(axis=0):\n',
#       np.repeat(data, repeats=3, axis=0)) # axis=0 행전체가 복사되니까 위아래로 쌓임!!!
#
# print('repeat(axis=1):\n',
#       np.repeat(data, repeats=3, axis=1)) # axis=1 열전체가 복사되니까 오른쪽으로..
#
# print('repeat(axis=0 and axis=1):\n',
#       np.repeat(np.repeat(data, repeats=2, axis=0),
#                 repeats=3, axis=1))

# -----------------------------------------------------------------------------------
# # np.tile > 전체
#
# data = np.arange(6).reshape(2,3)
# print(data)
#
# print('tile(axis=0):\n',
#       np.tile(data, reps=[3,1])) # axis=0 방향으로 3번 반복
#
# print('tile(axis=0:\n',
#       np.tile(data, reps=[1,3])) # axis=1 방향으로 3번 반복
#
# print('tile(axis=0 and axis=1:\n',
#       np.tile(data, reps=[3,3]))

# ---------------------------------------------------------------------
# check pattern image(8x8) > using np.tile
# 2x2짜리를 하나의 원소로 보기!

# white_patch = 255*np.ones(shape=(10,10))
# black_patch = 0*np.ones(shape=(10,10))
#
# img1 = np.hstack([white_patch, black_patch])
# img2 = np.hstack([black_patch, white_patch])
# img = np.vstack([img1, img2])
#
# check_pattern_image = np.tile(img, reps=[4,4])
#
# fig, ax = plt.subplots(figsize=(8,8))
# ax.imshow(check_pattern_image, cmap='gray')
#
# ax.tick_params(left=False, labelleft=False,
#                bottom=False, labelbottom=False)
#
# plt.show()

# ------------------------------------------------------------------------------------------

# white_patch = 255*np.ones(shape=(10,10))
# gray_patch = 128*np.ones(shape=(10,10))
# black_patch = 0*np.ones(shape=(10,10))
#
# img1 = np.hstack([white_patch, gray_patch])
# img2 = np.hstack([gray_patch, black_patch])
# img = np.vstack([img1, img2])
#
# check_pattern_image = np.tile(img, reps=[4,4])
#
# fig, ax = plt.subplots(figsize=(8,8))
# ax.imshow(check_pattern_image, cmap='gray')
#
# ax.tick_params(left=False, labelleft=False,
#                bottom=False, labelbottom=False)
#
# plt.show()

# -------------------------------------------------------------------------------

# img = np.arange(0, 256, 50).reshape(1,-1) # 0~255를 50씩 끊어서 색 가져옴
# img = img.repeat(100, axis=0).repeat(30, axis=1) # 좀 굵게 해주려고!
#
# fig, ax = plt.subplots(figsize=(8,4))
# ax.imshow(img, cmap='gray')
#
# ax.tick_params(left=False, labelleft=False,
#                bottom=False, labelbottom=False)
#
# fig.tight_layout()
# plt.show()

# --------------------------------------------------------------------------------

# img = np.arange(0, 151, 50).reshape(1,-1)  # 151인데도 흰색까지 다나옴 => cmap 사용하면 cmap에 끝점으로 알아서 설정해줌
# img = img.repeat(100, axis=0).repeat(30, axis=1)
#
# fig, ax = plt.subplots(figsize=(8,4))
# ax.imshow(img, cmap='gray')  # 151인데도 흰색까지 다나옴 => cmap 사용하면 cmap에 끝점으로 알아서 설정해줌
#
# ax.tick_params(left=False, labelleft=False,
#                bottom=False, labelbottom=False)
#
# fig.tight_layout()
# plt.show()

# -----------------------------------------------------------------------------------

# img = np.arange(0, 151, 50).reshape(1,-1) # 1행
# img = img.repeat(100, axis=0).repeat(30, axis=1)
#
# fig, ax = plt.subplots(figsize=(8,4))
# ax.imshow(img, cmap='gray', vmax=255, vmin=0)  # 이렇게 최대 최소를 지정 해주면 우리가 원하는 색을 잘 뽑아낼 수 있음!
#
# ax.tick_params(left=False, labelleft=False,
#                bottom=False, labelbottom=False)
#
# fig.tight_layout()
# plt.show()

# ---------------------------------------------------------------------------------------

# # gray_r
# img = np.arange(0, 256, 50).reshape(-1,1)  # 1열
# img = img.repeat(30, axis=0).repeat(100, axis=1)
#
# fig, ax = plt.subplots(figsize=(4,8))
# ax.imshow(img, cmap='gray_r', vmax=255, vmin=0) # 색 뒤집기 = 색_r
#
# ax.tick_params(left=False, labelleft=False,
#                bottom=False, labelbottom=False)
#
# fig.tight_layout()
# plt.show()

# -------------

# # step을 마이너스!
# img = np.arange(256, 0, -50).reshape(-1,1)  # 1열
# img = img.repeat(30, axis=0).repeat(100, axis=1)
#
# fig, ax = plt.subplots(figsize=(4,8))
# ax.imshow(img, cmap='gray', vmax=255, vmin=0) # 색 뒤집기 = 색_r
#
# ax.tick_params(left=False, labelleft=False,
#                bottom=False, labelbottom=False)
#
# fig.tight_layout()
# plt.show()

# -------------------

# # step을 마이너스! > 리스트 뒤집어주기[::-1]
# img = np.arange(0, 256, 50)[::-1].reshape(-1,1)  # 1열
# img = img.repeat(30, axis=0).repeat(100, axis=1)
#
# fig, ax = plt.subplots(figsize=(4,8))
# ax.imshow(img, cmap='gray', vmax=255, vmin=0) # 색 뒤집기 = 색_r
#
# ax.tick_params(left=False, labelleft=False,
#                bottom=False, labelbottom=False)
#
# fig.tight_layout()
# plt.show()

# --------------------

# # np.tile
# img = np.arange(0, 256, 50).reshape(-1,1)  # 1열
# img = np.tile(img, reps=[1,3])
#
# fig, ax = plt.subplots(figsize=(4,8))
# ax.imshow(img, cmap='gray_r', vmax=255, vmin=0) # 색 뒤집기 = 색_r
#
# ax.tick_params(left=False, labelleft=False,
#                bottom=False, labelbottom=False)
#
# fig.tight_layout()
# plt.show()

# --------------------------------------------------------------------------------------


# img1 = np.arange(0, 256, 2).reshape(1,-1)
# img2 = np.arange(0, 256, 2)[::-1].reshape(1,-1)
#
# img1 = img1.repeat(500, axis=0).repeat(10, axis=1)
# img2 = img2.repeat(500, axis=0).repeat(10, axis=1)
#
# img = np.vstack([img1,img2])
#
#
# fig, ax = plt.subplots(figsize=(4,4))
# ax.imshow(img, cmap='gray', vmax=255, vmin=0)
#
# ax.tick_params(left=False, labelleft=False,
#                bottom=False, labelbottom=False)
#
# fig.tight_layout()
# plt.show()






























