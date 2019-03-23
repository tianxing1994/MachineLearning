"""
通过自定义函数对数据预处理. 修改数据在不同区间的权重,
使得逻辑回归之后的分割线是曲线而不是直线.
曲线则能更好的拟合样本.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# def data_preprocess(data):
#     """得分 0.95"""
#     a = (data.iloc[:, 0])
#     b = (data.iloc[:, 1])
#     prod = a.mul(b)
#     result = prod.values.reshape(-1,1)
#     return result

# def data_preprocess(data):
#     """得分 0.97"""
#     a = data.iloc[:,0]
#     b = data.iloc[:,1]
#     result = (np.log2(a) * np.log10(b)).values.reshape(-1,1)
#     return result

# def data_preprocess(data):
#     """得分 1.0"""
#     a = data.iloc[:,0]
#     b = data.iloc[:,1]
#     x = 100 / (1 + np.power(1.14,(56-a)))
#     y = 100 / (1 + np.power(1.14,(56-b)))
#     result = pd.concat([x,y],axis=1)
#     return result

def data_preprocess(data):
    """得分 1.0"""
    a = data.iloc[:,0]
    b = data.iloc[:,1]
    x = 50 / (1 + np.power(1.07,(45-a))) + 50 / (1 + np.power(1.03,(81-a)))
    y = 50 / (1 + np.power(1.10,(44-b))) + 50 / (1 + np.power(1.02,(77-b)))
    result = pd.concat([x,y],axis=1)
    return result

pd_data = pd.read_csv(r"C:\Users\tianx\PycharmProjects\analysistest\dataset\LogiReg_data.txt", header=None, names=['Exam 1', 'Exam 2', 'Admitted'])

data =pd_data.iloc[:,:-1]
target = pd_data.iloc[:,-1]

logistic = LogisticRegression(solver="lbfgs")
logistic.fit(data_preprocess(data),target)
score = logistic.score(data_preprocess(data),target)
print(score)

x = np.linspace(data.iloc[:,0].min(),data.iloc[:,0].max(),1000)
y = np.linspace(data.iloc[:,1].min(),data.iloc[:,1].max(),1000)
X, Y = np.meshgrid(x,y)
X_test = np.c_[X.ravel(),Y.ravel()]

y_ = logistic.predict(data_preprocess(pd.DataFrame(X_test)))

figure = plt.figure(figsize=(1*8,1*6))

axis1 = figure.add_subplot(111)
axis1.pcolormesh(X,Y,y_.reshape(X.shape))
axis1.scatter(data.iloc[:,0],data.iloc[:,1],c=target,cmap='rainbow')
axis1.set_title("data_log")
plt.axis('equal')
plt.show()