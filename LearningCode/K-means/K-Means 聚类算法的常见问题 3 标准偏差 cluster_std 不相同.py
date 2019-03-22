"""
K-Means 聚类算法的常见问题 3
标准偏差 cluster_std 不相同

"""

from sklearn.cluster import KMeans
from sklearn.datasets import load_iris,make_blobs
import matplotlib.pyplot as plt
import numpy as np

X_train,y_train = make_blobs(n_samples=500, n_features=2, centers=3, cluster_std=[0.5,2,10])

kmeans = KMeans(n_clusters=3)
y_ = kmeans.fit_predict(X_train)

plt.figure(figsize=(12,5))
plt.subplot(121)
plt.scatter(X_train[:,0],X_train[:,1],c=y_train)
plt.subplot(122)
plt.scatter(X_train[:,0],X_train[:,1],c=y_)
plt.show()