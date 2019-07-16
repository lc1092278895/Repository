# -*- coding: -utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('harry.jpg', cv2.IMREAD_GRAYSCALE)  # 读取图片，
img1 = img.astype('float')  # 将uint8转化为float类型
img_dct = cv2.dct(img1)  # 进行离散余弦变换
img_dct_log = np.log(abs(img_dct))  # 进行log处理
img_recor = cv2.idct(img_dct)  # 进行离散余弦反变换
print(img_dct.shape)
zip_len=50
# 图片压缩，只保留100*100的数据
#recor_temp = img_dct[0:zip_len, 0:zip_len]
#recor_temp2 = np.zeros(img.shape)
#recor_temp2[0:zip_len, 0:zip_len] = recor_temp
recor_temp = img_dct[-300:, -300:]
recor_temp2 = np.zeros(img.shape)
recor_temp2[-300:, -300:] = recor_temp

# 压缩图片恢复
img_recor1 = cv2.idct(recor_temp2)

cv2.imshow("org",img);
cv2.waitKey();
cv2.imshow("dct",np.uint8(img_dct_log));
cv2.waitKey();
cv2.imshow("idct",np.uint8(img_recor));
cv2.waitKey();
cv2.imshow("idct2",np.uint8(img_recor1));
cv2.waitKey();
# 显示
'''
plt.subplot(221)
plt.imshow(img)
plt.title('original')

plt.subplot(222)
plt.imshow(img_dct_log)
plt.title('dct transformed')

plt.subplot(223)
plt.imshow(img_recor)
plt.title('idct transformed')

plt.subplot(224)
plt.imshow(img_recor1)
plt.title('idct transformed2')

plt.show()
'''