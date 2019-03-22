"""
通过 K-Means 算法, 对 data.txt 中的啤酒分类
并使用轮廓系数 silhouette_score 方法评估分类的好坏
"""

from sklearn.metrics import silhouette_score, silhouette_samples
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

beer_data = pd.read_csv(r"C:\Users\tianx\Desktop\code\data.txt",sep=" ")

# 数据中 name 列有英文文本, 是啤酒的名字,无用.  其它列皆为数字, 为 data.
data = beer_data.drop("name",axis=1)

kmeans = KMeans()
kmeans.fit(data)
labels = kmeans.labels_

# 查看分类的平均轮廓系数
avg_score = silhouette_score(data, labels)
# print(avg_score)

# 查看每个样本的轮廓系数
each_score = silhouette_samples(data, labels)
# print(each_score)

# 查看 KMeans 中 n_clusters 参数对轮廓系数的影响
scores = []
for k in range(2, 20):
    labels = KMeans(k).fit(data).labels_
    score = silhouette_score(data, labels)
    scores.append(score)
# print(scores)
