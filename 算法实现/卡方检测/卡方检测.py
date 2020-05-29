# coding=utf-8
"""
https://www.cnblogs.com/webRobot/p/6943562.html
"""
import matplotlib.pyplot as plt
from scipy.stats import norm, chi2
import numpy as np


def get_chisquare_data(n):
    # 标准正太分布
    normal_distribution = norm(0, 1)

    ret = list()
    for i in range(n):
        # 标准正态分布, 产生 30 个随机数.
        normal_data = normal_distribution.rvs(30)
        chisquare_data = normal_data ** 2
        ret.append(chisquare_data)
    return ret


def plot_chisquare(n):
    # 绘制自由度为 n 的卡方分布图, n 表示生成卡方数组的个数.
    list_data = get_chisquare_data(n)
    sum_data = sum(list_data)
    plt.hist(sum_data)
    plt.show()
    return


def demo1():
    plot_chisquare(2)
    plot_chisquare(3)
    plot_chisquare(10)
    return


def demo2():
    """绘制卡方分布. """
    fig, ax = plt.subplots(1, 1)

    # df 定义卡方的自由度.
    df = 20
    # mean, var, skew, kurt = chi2.stats(df, moments='mvsk')
    _, _, _, _ = chi2.stats(df, moments='mvsk')

    x = np.linspace(chi2.ppf(0.01, df), chi2.ppf(0.99, df), 100)

    ax.plot(x, chi2.pdf(x, df), 'r-', lw=5, alpha=0.6, label='chi2 pdf')
    plt.title("df is %d" % df)
    plt.show()
    return


if __name__ == '__main__':
    demo1()
    # demo2()
