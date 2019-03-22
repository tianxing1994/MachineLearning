"""
K-Means 聚类算法的常见问题 1
估计的中心点 n_clusters 值与真实值不相符.

"""

from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# 创建数据
iris = load_iris()
data = iris["data"][:,[2,3]]
target = iris["target"]

# 实例化模型
km = KMeans(n_clusters=4)
km.fit(data)

labels = km.labels_
centers = km.cluster_centers_

# 画图
figure = plt.figure(figsize=(12,5))
axis1 = plt.subplot(121)
plt.scatter(data[:,0],data[:,1],c=target,cmap="rainbow")
axis2 = plt.subplot(122)
plt.scatter(data[:,0],data[:,1],c=labels,cmap="rainbow")
plt.scatter(centers[:,0],centers[:,1],c=[0,1,2,3],cmap="rainbow",s=300,alpha=0.5)
plt.show()