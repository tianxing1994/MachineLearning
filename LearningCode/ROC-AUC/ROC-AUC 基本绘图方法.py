"""
主要函数: roc_curve(), auc()
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

pd_data = pd.read_csv(r"C:\Users\tianx\PycharmProjects\analysistest\dataset\LogiReg_data.txt", header=None, names=['Exam 1', 'Exam 2', 'Admitted'])

data = pd_data.iloc[:,[0,1]]
target = pd_data.iloc[:,2]

X_train, X_test, y_train, y_test = train_test_split(data,target,test_size=0.5)

logistic = LogisticRegression(solver="lbfgs")
logistic.fit(X_train,y_train)
y_ = logistic.predict_proba(X_test)

fpr, tpr, thresholds = roc_curve(y_test, y_[:,1], pos_label=1)

# auc 求曲线下的面积
auc_ = auc(fpr, tpr)

plt.plot(fpr, tpr, label=auc_)
plt.legend()
plt.show()