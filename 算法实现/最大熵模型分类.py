"""
https://blog.csdn.net/slx_share/article/details/80097020

https://github.com/Shi-Lixin/Machine-Learning-Algorithms
"""
import math
from collections import defaultdict

import numpy as np


class MaxEntry(object):
    def __init__(self, epsilon=1e-3, max_step=100):
        self._epsilon = epsilon
        self._max_step = max_step
        self._w = None
        self._labels = None
        self._feature_list = list()
        self._px = defaultdict(lambda: 0)
        self._pxy = defaultdict(lambda: 0)
        self._expect_feature = defaultdict(lambda: 0)
        self._data_list = list()
        self._n = None
        self._m = None
        self._n_feature = None

    def init_param(self, x_data, y_data):
        self._n = x_data.shape[0]
        self._labels = np.unique(y_data)
        self.feature_func(x_data, y_data)
        self._n_feature = len(self._feature_list)
        self._w = np.zeros(self._n_feature)
        self.expect_feature(x_data, y_data)
        return

    def feature_func(self, x_data, y_data):
        """
        特征函数.
        :param x_data:
        :param y_data:
        :return:
        """
        for x, y in zip(x_data, y_data):
            x = tuple(x)
            self._px[x] += 1.0 / self._n
            self._pxy[(x, y)] += 1.0 / self._n

            for i, value in enumerate(x):
                key = (i, value, y)
                if key not in self._feature_list:
                    self._feature_list.append(key)
        self._m = x_data.shape[1]
        return

    def expect_feature(self, x_data, y_data):
        """
        特征值的经验期望值.
        :param x_data:
        :param y_data:
        :return:
        """
        for x, y in zip(x_data, y_data):
            x = tuple(x)

            for i, value in enumerate(x):
                key = (i, value, y)
                self._expect_feature[key] += self._pxy[(x, y)]
        return

    def py_x(self, x):
        """
        跟据先验, 计算当前 x 属于各类别 y 的概率.
        :param x:
        :return:
        """
        py_x = defaultdict(float)
        for y in self._labels:
            s = 0
            for i, value in enumerate(x):
                key = (i, value, y)
                if key in self._feature_list:
                    s += self._w[self._feature_list.index(key)]
            py_x[y] = math.exp(s)
        normalizer = sum(py_x.values())
        for k, v in py_x.items():
            py_x[k] = v / normalizer
        return py_x

    def estimate_feature(self, x_data, y_data):
        estimate_feature = defaultdict(float)
        for x, y in zip(x_data, y_data):
            py_x = self.py_x(x)[y]
            x = tuple(x)
            for i, value in enumerate(x):
                key = (i, value, y)
                estimate_feature[key] += self._px[x] * py_x
        return estimate_feature

    def gis(self, x_data, y_data):
        estimate_feature = self.estimate_feature(x_data, y_data)
        delta = np.zeros(self._n_feature)
        for j in range(self._n_feature):
            delta[j] = 1 / self._m * math.log(
                self._expect_feature[self._feature_list[j]] / estimate_feature[self._feature_list[j]]
            )
            # try:
            #     delta[j] = 1 / self._m * math.log(
            #         self._expect_feature[self._feature_list[j]] / estimate_feature[self._feature_list[j]]
            #     )
            # except:
            #     continue
        delta = delta / delta.sum()
        return delta

    def iis(self, delta, x_data, y_data):
        g = np.zeros(self._n_feature)
        g_diff = np.zeros(self._n_feature)
        for j in range(self._n_feature):
            for k in range(self._n):
                x = tuple(x_data[k])
                g[j] += self._px[x] * self.py_x(x)[y_data[k]] * math.exp(delta[j] * self._m[k])
                g_diff[j] += g[j] * self._m[k]
            g[j] -= self._expect_feature[j]
            delta[j] -= g[j] / g_diff[j]
        return delta

    def fit(self, x_data, y_data):
        self.init_param(x_data, y_data)
        i = 0
        while i < self._max_step:
            i += 1
            delta = self.gis(x_data, y_data)
            self._w += delta
        return

    def predict(self, x):
        py_x = self.py_x(x)
        best_label = max(py_x, key=py_x.get)
        return best_label


def demo1():
    from sklearn.datasets import load_iris, load_digits
    from sklearn.model_selection import train_test_split

    data = load_iris()
    x_data = data['data']
    y_data = data['target']

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2)
    me = MaxEntry(max_step=10)
    me.fit(x_train, y_train)
    score = 0
    for x, y in zip(x_test, y_test):
        if me.predict(x) == y:
            score += 1
    print(score / len(y_test))
    return


if __name__ == '__main__':
    demo1()
