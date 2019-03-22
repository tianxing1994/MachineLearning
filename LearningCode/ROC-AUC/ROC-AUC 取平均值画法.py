"""
主要函数: roc_curve(), auc(), interp()
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from scipy import interp


pd_data = pd.read_csv(r"C:\Users\tianx\PycharmProjects\analysistest\dataset\LogiReg_data.txt", header=None, names=['Exam 1', 'Exam 2', 'Admitted'])

data =pd_data.iloc[:,[0,1]]
target = pd_data.iloc[:,2]

skf = StratifiedKFold(n_splits=4,shuffle=True,random_state=0)

fpr_mean = np.linspace(0,1,100)
tpr_mean = []

for train_index, test_index in skf.split(data,target):
    data_train = data.iloc[train_index]
    target_train = target.iloc[train_index]
    data_test = data.iloc[test_index]
    target_test = target.iloc[test_index]
    logistic = LogisticRegression(solver="lbfgs")
    logistic.fit(data_train, target_train)
    y_ = logistic.predict_proba(data_test)
    fpr, tpr, thresholds = roc_curve(target_test, y_[:, 1], pos_label=1)

    # 求 tpr_mean, 线性插值 interp() 函数, 根据已知的点自动预测指定点可能对应的值.
    tpr_m = interp(fpr_mean,fpr,tpr)
    tpr_mean.append(tpr_m)
    # auc 求曲线下面积
    auc_ = auc(fpr,tpr)
    plt.plot(fpr, tpr,label="auc: %0.3f" % auc_)

tpr_mean = np.array(tpr_mean).mean(axis=0)
auc_ = auc(fpr_mean,tpr_mean)
plt.plot(fpr_mean, tpr_mean, label="auc_mean: %0.3f" % auc_)

plt.legend()
plt.show()