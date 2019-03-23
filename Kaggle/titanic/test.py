"""
PassengerId: 乘客 ID
Survived: 得救
Pclass: 乘客等级(1/2/3等舱位)
Name: 姓名
Sex: 性别
Age: 年龄
SibSp: 堂兄弟/妹个数
Parch: 父母与小孩个数
Ticket: 船票信息
Fare: 票价
Cabin: 客舱
Embarked: 登船港口
"""

from sklearn.datasets import load_iris,make_blobs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression

# 读取数据
gender_submission = pd.read_csv(r"../../dataset/titanic/gender_submission.csv")
test_origin = pd.read_csv(r"../../dataset/titanic/test.csv")
train_origin = pd.read_csv(r"../../dataset/titanic/train.csv")


"""
# gender_submission.shape = (418,2)
# test.shape = (418,11)
# train.shape = (891,12)
# gender_submission.columns = Index(['PassengerId', 'Survived'], dtype='object')
# test.columns = Index(['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch','Ticket', 'Fare', 'Cabin', 'Embarked'],dtype='object')
# train.columns = Index(['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp','Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],dtype='object')

"""

#  Name 是无用信息, Sex, Ticket, Cabin, Embarked 是文本, 需要转化成数字.
test = test_origin.drop("Name",axis=1)
Sex_test_dict = dict(zip(test.loc[:,"Sex"].unique(),range(len(test.loc[:,"Sex"].unique()))))
Ticket_test_dict = dict(zip(test.loc[:,"Ticket"].unique(),range(len(test.loc[:,"Ticket"].unique()))))
Cabin_test_dict = dict(zip(test.loc[:,"Cabin"].unique(),range(len(test.loc[:,"Cabin"].unique()))))
Embarked_test_dict = dict(zip(test.loc[:,"Embarked"].unique(),range(len(test.loc[:,"Embarked"].unique()))))

test["Sex"] = test["Sex"].map(Sex_test_dict)
test["Ticket"]  = test["Ticket"].map(Ticket_test_dict)
test["Cabin"]  = test["Cabin"].map(Cabin_test_dict)
test["Embarked"]  = test["Embarked"].map(Embarked_test_dict)

train = train_origin.drop("Name",axis=1)
Sex_train_dict = dict(zip(train.loc[:,"Sex"].unique(),range(len(train.loc[:,"Sex"].unique()))))
Ticket_train_dict = dict(zip(train.loc[:,"Ticket"].unique(),range(len(train.loc[:,"Ticket"].unique()))))
Cabin_train_dict = dict(zip(train.loc[:,"Cabin"].unique(),range(len(train.loc[:,"Cabin"].unique()))))
Embarked_train_dict = dict(zip(train.loc[:,"Embarked"].unique(),range(len(train.loc[:,"Embarked"].unique()))))

train["Sex"] = train["Sex"].map(Sex_train_dict)
train["Ticket"]  = train["Ticket"].map(Ticket_train_dict)
train["Cabin"]  = train["Cabin"].map(Cabin_train_dict)
train["Embarked"]  = train["Embarked"].map(Embarked_train_dict)

# 合并 test 和 gender_submission
# test_data = pd.merge(test,gender_submission,how="inner",left_on="PassengerId",right_on="PassengerId")
# test_data.shape = (418,12)
# test_data.columns = Index(['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch','Ticket', 'Fare', 'Cabin', 'Embarked', 'Survived'],dtype='object')

"""
# 各列存在空数据的情况: (.any() 是否有空数据, .sum() 空数据的个数.)
# train.loc[:,"PassengerId"].isnull().any(): False, 0
# train.loc[:,"Survived"].isnull().any(): False, 0
# train.loc[:,"Pclass"].isnull().any(): False, 0
# train.loc[:,"Name"].isnull().any(): False, 0
# train.loc[:,"Sex"].isnull().any(): False, 0
# train.loc[:,"Age"].isnull().any(): True, 177
# train.loc[:,"SibSp"].isnull().any(): False, 0
# train.loc[:,"Parch"].isnull().any(): False, 0
# train.loc[:,"Ticket"].isnull().any(): False, 0
# train.loc[:,"Fare"].isnull().any(): False, 0
# train.loc[:,"Cabin"].isnull().any(): True, 687
# train.loc[:,"Embarked"].isnull().any(): True, 2

1. 合并 train 样本与 test 样本, 生成没有 Survived 字段的 all_feature_data 样本
2. 根据 all_feature_data 样本, 使用分类算法预测并填充 Embarked 的两个空值.
3. 使用线性回归预测填充 Age 中的 177 个空数据, 再取整填充.
4. Cabin 列, 空数据过多, 将空数据填充为"其它", 并判断是否有非空数据的准确率特别高的特征. 如没有 drop() 它
"""


# 合并 train 和 test 样本, 生成 all_feature_data 样本以预测年龄
train_drop_S = train.drop("Survived", axis=1)
all_feature_data = pd.concat([train_drop_S, test])
# 提取数据中年龄不为空, 与为空的数据. dwa: data_with_age, dwoa: data_without_age
data_with_age = all_feature_data[all_feature_data["Age"].notnull()]
data_without_age = all_feature_data[all_feature_data["Age"].isnull()]

dwa_data = data_with_age.drop("Age", axis=1)
dwa_target = data_with_age.loc[:, "Age"]
dwoa_data = data_without_age.drop("Age", axis=1)

# 逻辑回归
logstic = LogisticRegression()
logstic.fit(dwa_data,dwa_target)
dwoa_predict = logstic.predict(data_without_age)

print(dwoa_predict)
