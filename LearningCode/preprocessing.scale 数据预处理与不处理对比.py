"""
数据不进行预处理, 正则化处理, 以及
preprocessing.scale 函数数据预处理之后的区别比较.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression

def regularize(df):
    result = (df.max(axis=0) - df) / (df.max(axis=0) - df.min(axis=0))
    return result

pd_data = pd.read_csv(r"../dataset/LogiReg_data.txt", header=None, names=['Exam 1', 'Exam 2', 'Admitted'])

pd_data.insert(0,"Ones",1)
np_data = pd_data.values

data =np_data[:,1:3]
target = np_data[:,3]

logistic = LogisticRegression(solver="lbfgs")
logistic_regularized = LogisticRegression(solver="lbfgs")
logistic_scaled = LogisticRegression(solver="lbfgs")
logistic.fit(X=data,y=target)
logistic_regularized.fit(X=regularize(data),y=target)
logistic_scaled.fit(X=preprocessing.scale(data),y=target)

x = np.linspace(data[:,0].min(),data[:,0].max(),1000)
y = np.linspace(data[:,1].min(),data[:,1].max(),1000)
X, Y = np.meshgrid(x,y)
X_test = np.c_[X.ravel(),Y.ravel()]

y_ = logistic.predict(X_test)
y_regularized = logistic_regularized.predict(regularize(X_test))
y_scaled = logistic_scaled.predict(preprocessing.scale(X_test))

figure = plt.figure(figsize=(1*8,3*6))

# 未预处理的数据.
axis1 = figure.add_subplot(311)
axis1.pcolormesh(X,Y,y_.reshape(X.shape))
axis1.scatter(data[:,0],data[:,1],c=target,cmap='rainbow')
axis1.set_title("normal")

# regularize 函数处理过的数据.
axis2 = figure.add_subplot(312)
axis2.pcolormesh(X,Y,y_regularized.reshape(X.shape))
axis2.scatter(data[:,0],data[:,1],c=target,cmap='rainbow')
axis2.set_title("regularized")

# preprocessing.scale 函数处理过的数据
axis3 = figure.add_subplot(313)
axis3.pcolormesh(X,Y,y_scaled.reshape(X.shape))
axis3.scatter(data[:,0],data[:,1],c=target,cmap='rainbow')
axis3.set_title("scaled")

plt.show()