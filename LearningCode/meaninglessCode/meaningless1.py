from sklearn.svm import SVC, SVR
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

data, target = make_blobs(n_features=3,centers=2)


figure = plt.figure()
axis = Axes3D(figure)
axis.scatter(data[:,0],data[:,1],data[:,2],c=target,cmap="rainbow")

x = np.linspace(-10,10,100)
y = np.linspace(-10,10,100)
z = np.linspace(-10,10,100)

k = -0.10880958 * x + 0.09702804 * y + -0.08939903 * z - 1.93132476

X,Y = np.meshgrid(x,y)
Z = (-0.10880958 * X + 0.09702804 * Y + - 1.93132476) / 0.08939903


axis.plot_surface(X,Y,Z)

plt.show()

svc = SVC(kernel="linear")
svc.fit(data,target)

print(svc.coef_,svc.intercept_)

