"""训练数据中, 根据有 Age 值的样本, 线性回归预测出无 Age 值的样本之 Age 值. """

import re
from sklearn.linear_model import LinearRegression, LogisticRegression
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR

Titanic = pd.read_csv(r"C:\Users\tianx\PycharmProjects\analysistest\dataset\titanic\train.csv")
gender_submission = pd.read_csv(r"C:\Users\tianx\PycharmProjects\analysistest\dataset\titanic\gender_submission.csv")
Titanic_test = pd.read_csv(r"C:\Users\tianx\PycharmProjects\analysistest\dataset\titanic\test.csv")



# print(Titanic.isnull().any())
# print(Titanic.dtypes)

# Age, Cabin, Embarked 列有空值
# Name, Sex, Ticket, Cabin, Embarked 为文本值.

# print(len(Titanic))
# print(Titanic.loc[:,"Age"].isnull().sum())
# 训练样本有 891 个, Age 缺失值 177 个. 通过已有值, 线性回归求未知值.

# print(Titanic.loc[:,"Cabin"].isnull().sum())
# Cabin 缺失值 687 个, 大部份都已缺失, drop 掉.

# print(Titanic.loc[:,"Embarked"].isnull().sum())
# print(Titanic.loc[:,"Embarked"].value_counts())
# Embarked 只缺失 2 个, 该列只有三个值, S 644, C 168, Q 77. 用 S 来填充两个缺失值.

Titanic_1 = Titanic.drop("Cabin",axis=1)
Titanic_1 = Titanic_1.drop("Ticket",axis=1)
Titanic_1.loc[:,"Embarked"] = Titanic_1.loc[:,"Embarked"].fillna("S")
# print(Titanic_1.columns)
# print(Titanic_1.loc[:,"Embarked"].isnull().sum())

# 将所有字符串的列转换为数字.
object_columns = ["Sex", "Embarked"]
global_namespace = globals()
for column in object_columns:
    global_namespace[column] = dict(zip(Titanic_1.loc[:,column].unique(), range(len(Titanic_1.loc[:,column].unique()))))
    Titanic_1.loc[:, column] = Titanic_1.loc[:,column].map(global_namespace[column])

# print(Titanic_1.dtypes)
# 只剩 Name 列为 object 类型. 获取 Name 列各人的称谓.
Titanic_1.loc[:,"Name"] = Titanic_1.loc[:,"Name"].map(lambda x:re.search(" ([A-Za-z]+)\.", x)[0])
# 对转换为称谓后的 Name 列进行 Object 转 int64
Name_dict = dict(zip(Titanic_1.loc[:,"Name"].unique(), range(len(Titanic_1.loc[:,"Name"].unique()))))
Titanic_1.loc[:,"Name"] = Titanic_1.loc[:,"Name"].map(Name_dict)
# print(Titanic_1.dtypes)
# 类型转换完成.

# 取出 Age 中不为空的与为空的样本.
age_isnull = Titanic_1.loc[Titanic_1.loc[:,"Age"].isnull()]
age_notnull = Titanic_1.loc[Titanic_1.loc[:,"Age"].notnull()]

# print(age_isnull.loc[:,"Age"].isnull().sum())
# print(age_notnull.loc[:,"Age"].notnull().sum())
# isnull 样本 177 个, notnull 样本 714 个.

# 使用 SVR 线性回归.
# svr = SVR()
# svr.fit(age_notnull.drop("Age",axis=1),age_notnull.loc[:,"Age"])
# score_svr = svr.score(age_notnull.drop("Age",axis=1),age_notnull.loc[:,"Age"])
# print(score_svr)
# 得分 0.093

# parameters = {
#     # "kernel": ["linear","rbf","poly","sigmoid"],
#     "kernel": ["linear"],
#     'C':[2,]
# }
#
# svr = SVR(gamma="scale")
# clf = GridSearchCV(svr,parameters,cv=5)
# clf.fit(age_notnull.drop("Age",axis=1),age_notnull.loc[:,"Age"])
# print(clf.score(age_notnull.drop("Age",axis=1),age_notnull.loc[:,"Age"]))
# print(clf.best_estimator_)
# print(clf.best_score_)
# C=1.0, kernel="linear" 最佳得分 0.20

# 使用 SVR 线性回归.
linearR = LinearRegression()
linearR.fit(age_notnull.drop(["Age","Survived"],axis=1),age_notnull.loc[:,"Age"])
# score_linearR = linearR.score(age_notnull.drop("Age",axis=1),age_notnull.loc[:,"Age"])
# print(score_linearR)
# 得分 0.276

# 填充空数据.
age_pred = linearR.predict(age_isnull.drop(["Age","Survived"],axis=1))
age_isnull.loc[:,"Age"] = age_pred
# print(age_isnull.isnull().any())
# print(age_pred)
train_data = age_isnull.append(age_notnull)
# print(train_data.shape)

# 训练模型
logistic = LogisticRegression()
logistic.fit(train_data.drop("Survived",axis=1), train_data.loc[:,"Survived"])

# 检查测试数据
# print(Titanic_test.dtypes)
# print(Titanic_test.isnull().any())
# Name, Sex, Ticket, Embarked 为 object 类型
# Age, Fare, Cabin 存在缺失值

Titanic_test_1 = Titanic_test.drop("Cabin",axis=1)
Titanic_test_1 = Titanic_test_1.drop("Ticket",axis=1)
test_data_object_columns = ["Sex", "Embarked"]
for column in test_data_object_columns:
    Titanic_test_1.loc[:, column] = Titanic_test_1.loc[:,column].map(global_namespace[column])

Name_test_unique = Titanic_test_1.loc[:,"Name"].map(lambda x:re.search(" ([A-Za-z]+)\.", x)[0]).unique()

# for name in Name_test_unique:
#     if name not in Name_dict:
#         print(name)

# Dona. 不在 Name_dict 中.
# print(Name_dict)
Name_dict["Dona."] = 2
Titanic_test_1.loc[:,"Name"] = Titanic_test_1.loc[:,"Name"].map(lambda x:re.search(" ([A-Za-z]+)\.", x)[0])
Titanic_test_1.loc[:,"Name"] = Titanic_test_1.loc[:,"Name"].map(Name_dict)


Titanic_test_1.loc[:,"Name"].fillna(1,inplace=True)
Titanic_test_1.loc[:,"Fare"].fillna(method='ffill',inplace=True)
# print(Titanic_test_1.loc[:,"Age"].isnull().sum())

Titanic_test_age_isnull = Titanic_test_1.loc[Titanic_test_1.loc[:,"Age"].isnull()]
Titanic_test_age_notnull = Titanic_test_1.loc[Titanic_test_1.loc[:,"Age"].notnull()]

age_test_pred = linearR.predict(Titanic_test_age_isnull.drop("Age",axis=1))
Titanic_test_age_isnull.loc[:,"Age"] = age_test_pred
test_data = Titanic_test_age_isnull.append(Titanic_test_age_notnull)
# print(test_data.dtypes)
# print(test_data.isnull().any())

result_dict = dict(zip(gender_submission.loc[:,"PassengerId"], gender_submission.loc[:,"Survived"]))
test_target = test_data.loc[:,"PassengerId"].map(result_dict)
test_score = logistic.score(test_data, test_target)

print(test_score)
# 测试数据得分 95%

