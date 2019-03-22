import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold
from scipy import interp

credit = pd.read_csv(r'C:\Users\tianx\PycharmProjects\analysistest\dataset\creditcard.csv')
credit_data = credit.iloc[:,:-1]
credit_target = credit.iloc[:,-1]

# 正例, 被盗刷. 负例, 正常消费
# print(credit.iloc[:,-1].value_counts())
# 样本严重不平衡, 正例过少.
# 忽略一切不利情况, 直接使用逻辑回归, 查看得分与 AUC 线下面积. 采用 StratifiedKFold 循环预测, 求平均值.

skf = StratifiedKFold(n_splits=5)

fpr_mean = np.linspace(0,1,100)
tpr_mean = []

for train_index, test_index in skf.split(credit_data,credit_target):
    X_train = credit_data.iloc[train_index]
    y_train = credit_target.iloc[train_index]
    X_test = credit_data.iloc[test_index]
    y_test = credit_target.iloc[test_index]

    logistic = LogisticRegression()
    logistic.fit(X_train,y_train)
    y_predict = logistic.predict_proba(X_test)
    score = logistic.score(X_test,y_test)

    fpr, tpr, thresholds = roc_curve(y_test, y_predict[:,1], pos_label=1)

    tpr_m = interp(fpr_mean,fpr,tpr)
    tpr_mean.append(tpr_m)

    auc_ = auc(fpr,tpr)
    plt.plot(fpr,tpr,label="auc: %0.2f, score: %0.2f" % (auc_,score))

tpr_mean = np.array(tpr_mean).mean(axis=0)
auc_ = auc(fpr_mean,tpr_mean)
plt.plot(fpr_mean,tpr_mean,label="auc: %0.2f" % auc_)

plt.legend()
plt.show()







