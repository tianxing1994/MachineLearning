"""三种 SVC 分类, 与 LinearSVC 分类结果边界线对比"""

from sklearn.svm import SVC, SVR
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.svm import LinearSVC

# 创建数据
iris = load_iris()
data = iris["data"][:,[2,3]]
target = iris["target"]

# 创建预测模型
svc_linear = SVC(kernel='linear',gamma="auto")
svc_rbf = SVC(kernel='rbf',gamma="auto")
svc_poly = SVC(kernel='poly',gamma="auto")
linearsvc = LinearSVC()

# 训练模型
svc_linear.fit(data,target)
svc_rbf.fit(data,target)
svc_poly.fit(data,target)
linearsvc.fit(data,target)

# 生成网格
x = np.linspace(data[:,0].min(),data[:,0].max(),300)
y = np.linspace(data[:,1].min(),data[:,1].max(),300)
X,Y = np.meshgrid(x,y)
XY = np.c_[X.ravel(),Y.ravel()]

# 预测数据
svc_linear_Z = svc_linear.predict(XY)
svc_rbf_Z = svc_rbf.predict(XY)
svc_poly_Z = svc_poly.predict(XY)
linearsvc_Z = linearsvc.predict(XY)

# 绘制图像
figure = plt.figure(figsize=(2*5,2*5))
axis1 = figure.add_subplot(221)
axis1.pcolormesh(X,Y,svc_linear_Z.reshape(X.shape))
axis1.scatter(data[:,0],data[:,1],c=target,cmap="rainbow")
axis1.set_title("svc_linear")

axis2 = figure.add_subplot(222)
axis2.pcolormesh(X,Y,svc_rbf_Z.reshape(X.shape))
axis2.scatter(data[:,0],data[:,1],c=target,cmap="rainbow")
axis2.set_title("svc_rbf")

axis3 = figure.add_subplot(223)
axis3.pcolormesh(X,Y,svc_poly_Z.reshape(X.shape))
axis3.scatter(data[:,0],data[:,1],c=target,cmap="rainbow")
axis3.set_title("svc_poly")

axis4 = figure.add_subplot(224)
axis4.pcolormesh(X,Y,linearsvc_Z.reshape(X.shape))
axis4.scatter(data[:,0],data[:,1],c=target,cmap="rainbow")
axis4.set_title("linearsvc")

plt.show()