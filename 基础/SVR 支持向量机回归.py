"""
kernel: 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed' or a callable.
linear: ax + by + cz + d = 0
ploy: 多项式（polynomial）是指由变量、系数以及它们之间的加、减、乘、幂运算（非负整数次方）得到的表达式。
rbf:
sigmoid:
"""

from sklearn.svm import SVC, SVR
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.svm import LinearSVC

# 创建数据
x = np.linspace(-10,10,100).reshape(-1,1)
y = np.sin(x)
y += np.random.rand(*y.shape)

# 创建模型
SVR_linear = SVR(kernel="linear")
SVR_poly = SVR(kernel="poly")
SVR_rbf = SVR(kernel="rbf")
SVR_sigmoid = SVR(kernel="sigmoid")
# SVR_precomputed = SVR(kernel="precomputed")

# 训练模型
SVR_linear.fit(x,y)
SVR_poly.fit(x,y)
SVR_rbf.fit(x,y)
SVR_sigmoid.fit(x,y)
# SVR_precomputed.fit(x,y)

# 创建预测数据
a = np.linspace(-15,15,300).reshape(-1,1)

# 预测数据
svr_linear_b = SVR_linear.predict(a)
svr_poly_b = SVR_poly.predict(a)
svr_rbf_b = SVR_rbf.predict(a)
svr_sigmoid_b = SVR_sigmoid.predict(a)
# svr_precomputed_b = SVR_precomputed.predict(a)

plt.scatter(x,y)
plt.plot(a,svr_linear_b)
plt.plot(a,svr_poly_b)
plt.plot(a,svr_rbf_b)
plt.plot(a,svr_sigmoid_b)
plt.ylim(-3,3)
plt.xlim(-20,20)
plt.legend(['line','polynomial','radius','sigmoid'])
plt.show()
