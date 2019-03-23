from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeRegressor

iris = load_iris()
data = iris.data
target = iris.target

dt = DecisionTreeRegressor()
score = dt.fit(data,target).score(data,target)
# 一棵决策树, 发生过拟合.
print(score)

rf = RandomForestClassifier(n_estimators=10)
score = rf.fit(data,target).score(data,target)
print(score)








