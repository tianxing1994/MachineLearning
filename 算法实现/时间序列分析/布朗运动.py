# -*- coding: utf-8 -*-
from collections import deque
import random
import matplotlib.pyplot as plt
import numpy as np


def brownian_movement(mu=0, sigma=1):
    x = 0
    yield x

    while True:
        z = random.gauss(mu=mu, sigma=sigma)
        x += z
        yield x


def demo1():
    """随机游走, 布朗运动. """
    generator = brownian_movement(mu=0, sigma=1)
    ts = list()
    for i in range(1000000):
        ts.append(next(generator))

    plt.plot(ts)
    plt.show()
    return


def demo2():
    """从布朗运动序列求系统随机项的期望与方差. """
    generator = brownian_movement(mu=0, sigma=1)
    ts = list()
    for i in range(1000000):
        ts.append(next(generator))
    x = np.array(ts[100:-100], dtype=np.float64)

    x_delta = x[:-1] - x[1:]
    e_x_delta = np.mean(x_delta)
    d_x_delta = np.sqrt(np.mean(np.square(x_delta - e_x_delta)))
    print(f"E(Z): {e_x_delta}, D(Z): {d_x_delta}")
    return


if __name__ == '__main__':
    demo1()
    # demo2()
