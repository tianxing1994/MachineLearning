"""
采用 TfidfVectorizer 词向量特征提取的方法. 提取文本中具有特征的词汇,
并将这些词汇映射为可用于模型训练的数字矩阵.
再采用朴素贝叶斯预测文档的分类.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import TfidfVectorizer

sms = pd.read_csv(r'C:\Users\tianx\PycharmProjects\analysistest\dataset\emailSpamClassification\SMSSpamCollection',sep='\t',header=None)

target = sms.loc[:,0]
data = sms.iloc[:,1]

tf = TfidfVectorizer()
tf.fit(data)
tf_data = tf.transform(data)

X_train, X_test, y_train, y_test = train_test_split(tf_data, target,test_size=0.1)

b_NB = BernoulliNB()
b_NB.fit(X_train,y_train)
score = b_NB.score(X_test,y_test)

print(score)




