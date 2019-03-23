"""使用逻辑回归, 得分 95%"""

from sklearn.datasets import load_iris,make_blobs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression

train_data = pd.read_csv(r"C:\Users\tianx\PycharmProjects\analysistest\dataset\titanic\train_data.csv")
data_train = train_data.drop(["PassengerId","Survived","Name"],axis=1)
target_train = train_data.loc[:,"Survived"]

test_data = pd.read_csv(r"C:\Users\tianx\PycharmProjects\analysistest\dataset\titanic\test_data.csv")
data_test = test_data.drop(["PassengerId","Survived","Name"],axis=1)
target_test = test_data.loc[:,"Survived"]

# SVC 分类算法
# svc = SVC()
# svc.fit(data_train,target_train)
# svc_score = svc.score(data_test,target_test)
# print(svc_score)
# 测试数据得分 0.714

# LogisticRegression 回归算法
logistic = LogisticRegression()
logistic.fit(data_train,target_train)
logistic_score = logistic.score(data_test,target_test)
print(logistic_score)
# 测试数据得分 0.9496402877697842

