"""
K-Means 聚类算法的常见问题 2
数据存在偏差

"""

from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import numpy as np


# 创建数据
iris = load_iris()
data = iris["data"][:,[2,3]]
trans = [[0.6,-0.6],[-0.4,0.8]]
data_trans = np.dot(data, trans)
target = iris["target"]

# 实例化模型
n_clusters = 3
km = KMeans(n_clusters=n_clusters)
km.fit(data_trans)

labels = km.labels_
centers = km.cluster_centers_

# 画图
figure = plt.figure(figsize=(12,5))
axis1 = plt.subplot(121)
plt.scatter(data_trans[:,0],data_trans[:,1],c=target,cmap="rainbow")
axis2 = plt.subplot(122)
plt.scatter(data_trans[:,0],data_trans[:,1],c=labels,cmap="rainbow")
plt.scatter(centers[:,0],centers[:,1],c=range(n_clusters),cmap="rainbow",s=300,alpha=0.5)
plt.show()