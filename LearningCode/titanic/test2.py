import re

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
# train_1 = train_origin.fillna("S")
print(train_1)
# train_2 = train_1.drop("Cabin",axis=1)
# test_2 = test_origin.drop("Cabin",axis=1)









