"""
我看了一些资料, 并参考了
https://blog.csdn.net/slx_share/article/details/80097020
之后, 对最大熵模型的理解, 以代码实现如下.
我的理解在 最大熵模型.md. 用 Typora 打开.
我的理解中不合理的地方很多, 没办法, 还没完全理解透.
"""
import math
from collections import defaultdict

import numpy as np


class MaxEntry(object):
    def __init__(self, epsilon=1e-3, max_step=1000, learning_rate=1e-3):
        self._epsilon = epsilon
        self._max_step = max_step
        self._learning_rate = learning_rate
        self._w = None
        self._labels = None
        self._feature_list = list()
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
        根据当前模型参数 self._w, 计算 x 发生时, y 发生的概率
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
            # 由于 s 的最从 0 开始迭代, 所以一开始值很小时出现值无效, 除以 0 等问题. 所以映射成 e^{x}.
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
                estimate_feature[key] += py_x
        return estimate_feature

    def gis(self, x_data, y_data):
        estimate_feature = self.estimate_feature(x_data, y_data)
        delta = np.zeros(self._n_feature)
        for j in range(self._n_feature):
            try:
                delta[j] = self._learning_rate * math.log(
                    self._expect_feature[self._feature_list[j]] / estimate_feature[self._feature_list[j]]
                )
            except:
                # 在训练集中所有的 (i, value, y) 特征一定都可以索引得到, 但对没有见过的测试集, 则索引可能报错.
                continue
        delta = delta / delta.sum()
        return delta

    def fit(self, x_data, y_data):
        self.init_param(x_data, y_data)
        i = 0
        while i < self._max_step:
            i += 1
            delta = self.gis(x_data, y_data)
            self._w += delta
            max_delta = max(delta)
            if max_delta < self._epsilon:
                break
        print("end training")
        print(f"steps: {i}")
        print(f"finally max delta: {max_delta}")
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
