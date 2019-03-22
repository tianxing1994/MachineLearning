"""
K-Means 聚类算法的常见问题 4
数据集中各分类的样本差距很大
"""

from sklearn.cluster import KMeans
from sklearn.datasets import load_iris,make_blobs
import matplotlib.pyplot as plt
import numpy as np

# 创建训练集, 控制 center_box, 使小样本集相互有所交错, 大样本集独立开
x1, y1 = make_blobs(n_samples=1000, n_features=2, centers=1,center_box=(-6,10))
x2, y2 = make_blobs(n_samples=200, n_features=2, centers=1,center_box=(-8,-2))
x3, y3 = make_blobs(n_samples=50, n_features=2, centers=1,center_box=(-6,-2))
x = np.concatenate((x1,x2,x3))
y = [0] * 1000 + [1] * 200 + [2] * 50

kmeans = KMeans(n_clusters=3)
kmeans.fit(x)

labels = kmeans.labels_

plt.figure(figsize=(12,5))
plt.subplot(121)

plt.scatter(x[:,0],x[:,1],c=y,cmap="rainbow")
plt.subplot(122)
plt.scatter(x[:,0],x[:,1],c=labels,cmap="rainbow")

plt.show()