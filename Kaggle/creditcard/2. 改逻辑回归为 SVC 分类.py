import numpy as np
import pandas as pd
from pandas import Series,DataFrame
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold
from scipy import interp
from sklearn.svm import SVC

credit = pd.read_csv(r'C:\Users\tianx\PycharmProjects\analysistest\dataset\creditcard.csv')
credit_data = credit.iloc[:,:-1]
credit_target = credit.iloc[:,-1]
# 正例, 被盗刷. 负例, 正常消费

skf = StratifiedKFold(n_splits=5)

fpr_mean = np.linspace(0,1,100)
tpr_mean = []

for train_index, test_index in skf.split(credit_data,credit_target):
    X_train = credit_data.iloc[train_index]
    y_train = credit_target.iloc[train_index]
    X_test = credit_data.iloc[test_index]
    y_test = credit_target.iloc[test_index]

    svc = SVC()
    svc.fit(X_train,y_train)
    y_predict = svc.predict_proba(X_test)
    score = svc.score(X_test,y_test)

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
