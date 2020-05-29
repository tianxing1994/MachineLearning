# coding=utf-8
"""
AR 自回归时间序列模型.
X_{t} = \phi_{1} X_{t-1} + \phi_{2} X_{t-2} + \phi_{3} X_{t-3} + \cdots + \phi_{k} X_{t-k} + Z_{t}

MA 滑动平均模型.
X_{t} = Z_{t} + \phi_{1} Z_{t-1} + \phi_{2} Z_{t-2} + \phi_{3} Z_{t-3} + \cdots + \phi_{k} Z_{t-k}

"""
from collections import deque
import random
import matplotlib.pyplot as plt


def white_noise(mu=0, sigma=1):
    while True:
        z = random.gauss(mu=mu, sigma=sigma)
        yield z


def auto_regression_p(phi_list):
    history_x_list = deque(maxlen=len(phi_list))
    # z = random.normalvariate(mu=0, sigma=1)

    while True:
        z = random.gauss(mu=0, sigma=1)
        x = 0
        for phi, history_x in zip(phi_list, reversed(history_x_list)):
            x += phi * history_x
        x += z
        history_x_list.append(x)
        yield x


def moving_average_p(phi_list):
    history_z_list = deque(maxlen=len(phi_list))

    while True:
        z = random.gauss(mu=0, sigma=1)
        x = 0
        for phi, history_z in zip(phi_list, reversed(history_z_list)):
            x += phi * history_z
        x += z
        history_z_list.append(z)
        yield x


def demo1():
    """
    在本例中, phi 值是我随便填写的, 结果不收敛.
    也就是说, 对于平隐序列, AR 模型虽然是表达式那样, 但对于 phi 值一定还有一些限定.

    要求特征多项式的特征方程的复数根的模长大于 1

    必要但不充分条件是:
    phi_{1} + phi_{2} + phi_{3} + ... + phi_{p} < 1
    |phi_{p}| < 1
    """
    phi_list = [0.9, 0.7, 0.5, 0.3, 0.1]
    generator = auto_regression_p(phi_list)
    ts = list()
    for i in range(100):
        ts.append(next(generator))

    plt.plot(ts)
    plt.show()
    return


def demo2():
    """
    自回归模型 x_{t} = x_{t-1} - 0.8 * x_{t-2} + w_{t}
    phi_{1} = 1, phi_{2} = -0.8
    """
    phi_list = [1, -0.8]
    generator = auto_regression_p(phi_list)
    ts = list()
    for i in range(3000):
        ts.append(next(generator))

    plt.plot(ts)
    plt.show()
    return


def demo3():
    """白噪声"""
    generator = white_noise()
    ts = list()
    for i in range(3000):
        ts.append(next(generator))
    plt.plot(ts)
    plt.show()
    return


def demo4():
    """滑动平均模型"""
    phi_list = [0.36, 0.85]
    generator = moving_average_p(phi_list)
    ts = list()
    for i in range(3000):
        ts.append(next(generator))

    plt.plot(ts)
    plt.show()
    return


if __name__ == '__main__':
    # demo1()
    # demo2()
    # demo3()
    demo4()
