from sklearn.svm import SVC, SVR
from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV

iris = load_iris()

parameters = {
    "kernel": ["linear", "rbf", "poly"],
    'C': [1, 5, 10]
}

svc = SVC(gamma="scale")
clf = GridSearchCV(svc,parameters,cv=5)
clf.fit(iris.data,iris.target)
print(clf.best_estimator_)
