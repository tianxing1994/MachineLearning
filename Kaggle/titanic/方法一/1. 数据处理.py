"""数据预处理并保存到 CSV 文件"""

import re

import pandas as pd
from sklearn.svm import SVR

# 读取数据
gender_submission = pd.read_csv(r"C:\Users\tianx\PycharmProjects\analysistest\dataset\titanic\gender_submission.csv")
test_origin = pd.read_csv(r"C:\Users\tianx\PycharmProjects\analysistest\dataset\titanic\test.csv")
train_origin = pd.read_csv(r"C:\Users\tianx\PycharmProjects\analysistest\dataset\titanic\train.csv")

# 字段解释
# PassengerId: 乘客 ID
# Survived: 得救
# Pclass: 乘客等级(1/2/3等舱位)
# Name: 姓名
# Sex: 性别
# Age: 年龄
# SibSp: 堂兄弟/妹个数
# Parch: 父母与小孩个数
# Ticket: 船票信息, 编号
# Fare: 票价
# Cabin: 客舱, 舱位
# Embarked: 登船港口

# test_origin:
# 类型: <class 'pandas.core.frame.DataFrame'>, 形状: (418,11),
# 列名: ['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch','Ticket', 'Fare', 'Cabin', 'Embarked']
# 存在字符串的字段: Name, Sex, Ticket, Cabin, Embarked
# 存在为空的字符: Age: 86个, Fare: 1个, Cabin: 327个,

# train_origin:
# 类型: <class 'pandas.core.frame.DataFrame'>, 形状: (891,12),
# 列名: ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp','Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
# 存在字符串的字段: Name, Sex, Ticket, Cabin, Embarked
# 存在为空的字符: Age: 177个, Cabin: 687个, Embarked: 2个.


# 查看 train_origin Embarked 列, S: 644个, C: 168个, Q: 77个. Embarked 只缺了两个, 用最多的 S 去填充它.
train_origin.loc[:,"Embarked"] = train_origin.loc[:,"Embarked"].fillna("S")
train_1 = train_origin
# 查看 train_1 Embarked 列, S: 646个, C: 168个, Q: 77个.
# print(train_1.loc[:,"Embarked"].isnull().any())
# print(train_1.loc[:,"Embarked"].value_counts())

# Cabin 缺得太多. drop 掉. train_2.shape = (891,11), test_2.shape = (418,10)
train_2 = train_1.drop("Cabin",axis=1)
test_2 = test_origin.drop("Cabin",axis=1)

# drop 掉 train_2 中的 Survived 列, 再合并 train 和 test.
# train_test.shape = (1309,10)
# train_3 = train_2.drop("Survived",axis=1)
train_3 = train_2
train_test = pd.concat([train_3,test_2],axis=0,sort=False)

# Ticket 船票编号没用, drop 掉
train_test_1 = train_test.drop("Ticket",axis=1)

# 将所有的字符转成数字. 根据 train_test 生成映射字典
# train_test_Sex_dict: {'male': 0, 'female': 1}, train_test_Embarked_dict: {'S': 0, 'C': 1, 'Q': 2}
train_test_Sex_dict = dict(zip(train_test_1.loc[:,"Sex"].unique(),range(len(train_test_1.loc[:,"Sex"].unique()))))
train_test_Embarked_dict = dict(zip(train_test_1.loc[:,"Embarked"].unique(),range(len(train_test_1.loc[:,"Embarked"].unique()))))

# 处理名字. 1. 增加头衔字段. 并创建映射字典
# train_test_Title_dict: {' Mr.': 0, ' Mrs.': 1, ' Miss.': 2, ' Master.': 3, ' Don.': 4, ' Rev.': 5, ' Dr.': 6, ' Mme.': 7, ' Ms.': 8, ' Major.': 9, ' Lady.': 10, ' Sir.': 11, ' Mlle.': 12, ' Col.': 13, ' Capt.': 14, ' Countess.': 15, ' Jonkheer.': 16, ' Dona.': 17}

train_test_1.loc[:,"Title"] = train_test_1.loc[:,"Name"].map(lambda x:re.search(" ([A-Za-z]+)\.", x)[0])
train_test_Title_dict = dict(zip(train_test_1.loc[:,"Title"].unique(),range(len(train_test_1.loc[:,"Title"].unique()))))

# 处理字符串字段: Sex, Embarked, Title
train_test_1.loc[:,"Sex"] = train_test_1.loc[:,"Sex"].map(train_test_Sex_dict)
train_test_1.loc[:,"Embarked"] = train_test_1.loc[:,"Embarked"].map(train_test_Embarked_dict)
train_test_1.loc[:,"Title"] = train_test_1.loc[:,"Title"].map(train_test_Title_dict)

train_test_1.loc[:,"Survived"] = train_test_1.loc[:,"Survived"].fillna(2)
# Fare 列中有一个为空的. drop 掉. print(train_test_1.isnull().sum())
train_test_1 = train_test_1.loc[train_test_1.loc[:,"Fare"].notnull()]
# print(train_test_1.loc[:,"Fare"].value_counts())


# 提取 train_test_1 中 Age 不为空的和为空的两种表.
train_test_Age_isna = train_test_1.loc[train_test_1.loc[:,"Age"].isna()]
train_test_Age_notna = train_test_1.loc[train_test_1.loc[:,"Age"].notna()]

# print(train_test_Age_isna.columns)

svr = SVR(gamma='scale')
svr.fit(train_test_Age_notna.drop(["Age","Name"],axis=1),train_test_Age_notna.loc[:,"Age"])
train_test_Age_isna_predict = svr.predict(train_test_Age_isna.drop(["Age","Name"],axis=1))
# print(train_test_Age_isna_predict.shape)
# print(train_test_Age_isna_predict)

# 将预测出的年龄填充上去.
train_test_Age_isna.loc[:,"Age"] = train_test_Age_isna_predict
# print(train_test_Age_isna)



# 合并 train_test_Age_isna, train_test_Age_notna 数据
train_test_2 = train_test_Age_isna.append(train_test_Age_notna)

# 提取 Survived != 2 的数据为训练数据, Survived == 2 的数据为测试数据.
train_test_2_train = train_test_2.loc[(train_test_2.loc[:,"Survived"] != 2)]
train_test_2_test = train_test_2.loc[(train_test_2.loc[:,"Survived"] == 2)]


keys = gender_submission.loc[:,"PassengerId"]
values = gender_submission.loc[:,"Survived"]
map_dict = dict(zip(keys,values))

train_test_2_test.loc[:,"Survived"] = train_test_2_test.loc[:,"PassengerId"].map(map_dict)

# 保存 train_test_2_train, train_test_2_test 到本地.
train_test_2_train.to_csv(path_or_buf=r"C:\Users\tianx\PycharmProjects\analysistest\dataset\titanic\train_data.csv",index=False)
train_test_2_test.to_csv(path_or_buf=r"C:\Users\tianx\PycharmProjects\analysistest\dataset\titanic\test_data.csv",index=False)