from sklearn.cluster import KMeans
from sklearn.datasets import load_iris,make_blobs
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle

img = plt.imread(r"C:\Users\tianx\AppData\Roaming\feiq\Recv Files\day12\data\bird_small.png")
# img.shape = (128,128,3)

img_reshape = img.reshape(-1,3)
# img_reshape.shape = (16384,3)

# 原始数据量太大, 随机选取少量数据作训练
img_test = shuffle(img_reshape)[:1000]

# 将图片改成只有 32种颜色
kmeans = KMeans(n_clusters=32)
kmeans.fit(img_test)

# 预测数据
index = kmeans.predict(img_reshape)

centers = kmeans.cluster_centers_

new_img_data = centers[index]
# new_img_data.shape = (16384,3)

new_img = new_img_data.reshape(img.shape)

plt.figure(figsize=(12,5))
plt.subplot(121)
plt.imshow(img)
plt.subplot(122)
plt.imshow(new_img)
plt.show()