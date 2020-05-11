# coding=utf-8
"""
https://www.cnblogs.com/webRobot/p/6943562.html
"""
import matplotlib.pyplot as plt
from scipy.stats import norm, chi2
import numpy as np


n = 10


# 绘制自由度为 n 的卡方分布图, n 表示生成卡方数组的个数.
def get_chisquare_data(n):
    # 标准正太分布
    normal_distribution = norm(0, 1)

    ret = list()
    for i in range(n):
        normal_data = normal_distribution.rvs(30)
        print(normal_data)
        chisquare_data = normal_data ** 2
        ret.append(chisquare_data)
    return ret


def plot_chisquare(n):
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
    fig, ax = plt.subplots(1, 1)

    df = 20
    mean, var, skew, kurt = chi2.stats(df, moments='mvsk')

    # 绘制函数的起始点和终止点
    # pdf为概率密度函数
    # 百分比函数(PPF) :the inverse of the CDF. PPF  函数和连续分布函数CDF相逆，
    # 比如输入哪一个点，可以得到低于等于20的概率？
    # ppf(0.01, df)表示输入哪个点，得到概率低于0.01
    initial = chi2.ppf(0.01, df)
    end = chi2.ppf(0.99, df)
    x = np.linspace(initial, end, 100)

    # 概率密度函数用于绘图
    ax.plot(x, chi2.pdf(x, df), 'r-', lw=5, alpha=0.6, label='chi2 pdf')
    plt.title("df is %d" % df)
    plt.show()

    return


if __name__ == '__main__':
    # demo1()
    demo2()
