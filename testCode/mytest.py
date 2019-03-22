import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# def score_analysis_1(dataset,n=10):
#     columns_name = dataset.columns
#     data_sorted = dataset.sort_values(by=columns_name[0], axis=0)
#
#     l = len(dataset)
#     k = np.math.ceil(l / n)
#     scores_l = []
#     dataset_l = []
#
#     for i in range(k):
#         data_k = data_sorted.iloc[i * n:(i * n + n)]
#         score_k = logistic.score(data_k.iloc[:, :-1], data_k.iloc[:, -1])
#         scores_l.append(score_k)
#         dataset_l.append(data_k)
#
#     return scores_l, dataset_l

pd_data = pd.read_csv(r"../dataset/LogiReg_data.txt", header=None, names=['Exam 1', 'Exam 2', 'Admitted'])

data =pd_data.iloc[:,:-1]
X = data.iloc[:,0]
Y = data.iloc[:,1]
target = pd_data.iloc[:,-1]

logistic = LogisticRegression(solver="lbfgs")
logistic.fit(data,target)
score = logistic.score(data,target)

y_ = pd.Series(logistic.predict(data),name="y_")
error = pd.Series((target - y_),name="error")


result_data = pd.concat([data,target,y_,error],axis=1)
# print(result_data)


error_data=result_data.loc[result_data.loc[:,'error'] != 0]


# 画图
# X
result_data_sorted = result_data.sort_values(by="Exam 1", axis=0)
plt.plot(result_data_sorted.iloc[:,0],result_data_sorted.iloc[:,-1])

# Y
result_data_sorted = result_data.sort_values(by="Exam 2", axis=0)
plt.plot(result_data_sorted.iloc[:,1],result_data_sorted.iloc[:,-1])

plt.show()
