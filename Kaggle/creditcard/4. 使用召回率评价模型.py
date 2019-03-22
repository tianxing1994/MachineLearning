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
# Recall (召回率):
# 正例被正确预测出的比率.

X_train,X_test,y_train,y_test = train_test_split(credit_data,credit_target)

logistic = LogisticRegression()
logistic.fit(X_train,y_train)
y_predict = logistic.predict(X_test)
score = logistic.score(X_test,y_test)

print(score)

# Jupyter Notebook:
# pd.crosstab(index=y_predict,columns=y_test,rownames=['预测值'], colnames=['真实值'])
# 得到 TP: 81, FN: 58. 召回率 Recall: TP/(TP+FN) = 1.397
# 真实值	0	1
# 预测值
# 0	    71025	58
# 1	    38	81


