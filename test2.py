# coding=utf-8
from sklearn.feature_selection import chi2
from sklearn.datasets import load_iris


iris = load_iris()

chi2_ret, pval_ret = chi2(iris.data, iris.target)
print(chi2_ret)
print(pval_ret)
