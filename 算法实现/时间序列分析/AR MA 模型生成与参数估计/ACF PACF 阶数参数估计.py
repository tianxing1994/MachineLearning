# coding=utf-8
from collections import deque
import random
import matplotlib.pyplot as plt
import numpy as np


def white_noise(mu=0, sigma=1):
    while True:
        z = random.gauss(mu=mu, sigma=sigma)
        yield z


def auto_regression_q(phi_list, mu=0, sigma=1):
    history_x_list = deque(maxlen=len(phi_list))

    while True:
        z = random.gauss(mu=mu, sigma=sigma)
        x = 0
        for phi, history_x in zip(phi_list, reversed(history_x_list)):
            x += phi * history_x
        x += z
        history_x_list.append(x)
        yield x


def moving_average_q(phi_list, mu=0, sigma=1):
    history_z_list = deque(maxlen=len(phi_list))

    while True:
        z = random.gauss(mu=mu, sigma=sigma)
        x = 0
        for phi, history_z in zip(phi_list, reversed(history_z_list)):
            x += phi * history_z
        x += z
        history_z_list.append(z)
        yield x


def get_gamma_matrix(x, q):
    """
    计算时间序列的自相关函数矩阵.
    :param x: 时间序列
    :param q: 猜想的模型的阶数.
    :return: 返回 \gamma_{0} 到 \gamma_{k} 的方阵, 方阵大小为 (k + 1, k + 1).
    """
    x_t = list()
    n = len(x)
    for i in range(q):
        start = i
        end = -(q - i)
        x_i = x[start:end]
        x_t.append(x_i)
    else:
        x_i = x[q:]
        x_t.append(x_i)
    x_t = np.array(x_t)
    x_t_mean = np.mean(x_t, axis=1, keepdims=True)
    x_t_ = x_t - x_t_mean
    ret = np.dot(x_t_, x_t_.T) / n
    return ret


def acf(x, q):
    """
    ACF (AutoCovariance Function) 自相关函数.
    计算时间序列的自相关函数.
    :param x: 时间序列
    :param q: 猜想的模型的阶数.
    :return: 返回 \gamma_{0} 到 \gamma_{q} 的自相关函数. 总共有 q + 1 项.
    """
    x_mean = np.mean(x)
    # 采用以下方式修改 x 会将函数外部的 x 序列也改变. 因为 ndarray 传参是引用来实现的.
    # x -= x_mean
    ts = x - x_mean
    n = len(ts)
    gamma = list()
    for i in range(q+1):
        if i == 0:
            gamma_0 = np.sum(ts * ts) / n
            gamma.append(gamma_0)
        else:
            gamma_i = np.sum(ts[:-i] * ts[i:]) / n
            gamma.append(gamma_i)
    return gamma


def pacf(x, q):
    """
    PACF (Partial AutoCovariance Function) 偏自相关函数.
    :param x: 时间序列.
    :param q: 猜想的模型的阶数.
    :return: phi, gamma, Gamma.
    phi: \phi_{1} 到 \phi_{k} 的值.
    gamma: \gamma_{1} 到 \gamma_{k} 的值.
    Gamma: \gamma_{0} 到 \gamma_{k-1} 组成的自协方差矩阵.
    """
    gamma_matrix = get_gamma_matrix(x, q)
    gamma = gamma_matrix[0, 1:]
    Gamma = gamma_matrix[:-1, :-1]
    Gamma_i = np.linalg.inv(Gamma)
    phi = np.dot(Gamma_i, gamma)
    return phi, gamma, Gamma


def calc_ar_ez(x, phi):
    """从时间序列计算随机变量 Z 的期望. """
    ez = (1 - np.sum(phi)) * np.mean(x)
    return ez


def calc_ar_dz(gamma_0, gamma, phi):
    """从时间序列计算随机变量 Z 的方差. """
    ret = np.sqrt(gamma_0 - np.dot(phi, gamma))
    return ret


def calc_ma_phi_iter(gamma, phi_init):
    """
    通过线性迭代的方法求解 MA 模型的 phi 参数.
    这种方法可能会出现不收敛的情况.
    1. 这时可以考虑增大时间序列 x 的值的数量.
    2. 初始化 phi 参数应满足一定的条件, 可以理解 phi 参数其实也是一种衰减系数, 因此它应该绝对值小于 1.
    仅仅是绝对值小于 1, 还不够充分, 我想这里还需要仔细研究 phi 应该足的那个多项比方程. 我还没有理解它.
    也可以考虑用矩阵方法估计的结果作线性迭代方法的 phi 参数初始化.

    直接用 gamma 作为 phi 初始化参数效果还比较可以.
    :param gamma: \gamma_{0} 到 \gamma_{k} 的自相关函数.
    :param phi_init: phi 的初始化参数.
    :return: phi_prev, loss. phi_prev 最终估计出的 phi 参数. loss 最终的线性方程组的损失.
    """
    q = len(gamma) - 1
    sigma_square = 0
    phi_prev = np.array(phi_init, dtype=np.float64)
    phi_this = np.array([1, 0, 0], dtype=np.float64)
    for n_iter in range(100):
        sigma_square = gamma[0] / np.dot(phi_prev, phi_prev)
        for k in range(1, q):
            phi_this[k] = gamma[k] / sigma_square - np.dot(phi_prev[1:q-k+1], phi_prev[k+1:q+1])
        else:
            phi_this[q] = gamma[q] / sigma_square
        phi_prev = phi_this
        phi_this = np.array([1, 0, 0], dtype=np.float64)

    loss = 0
    for k in range(q + 1):
        loss += np.abs(gamma[k] - sigma_square * np.dot(phi_prev[0:q-k+1], phi_prev[k:q+1]))
    return phi_prev, loss


def calc_omega(x, q, k):
    """
    计算用矩阵法估计 MA 模型参数所需要的参数.
    :param x: 时间序列.
    :param q: 猜想的模型的阶数.
    :param k: 矩阵法估计 MA 模型参数中 omega 矩阵的 k 除数.
    omega 的形状为 (q, k), 每个位置 (i, j) 对应自相关函数 \gamma_{i+j+1}.
    :return: omega, Gamma_inv, gamma_0, gamma_1_q
    omega: ndarray, 状为 (q, k), 每个位置 (i, j) 对应自相关函数 \gamma_{i+j+1}.
    Gamma_inv: k 阶的自协方差矩阵的逆矩阵.
    gamma_0: \gamma_{0} 自相关函数.
    gamma_1_q: ndarray, 形状为 (q,)  \gamma_{1} 到 \gamma_{q} 的自相关函数组成.
    """
    gamma = acf(x, 2*k-1)
    omega = np.zeros(shape=(q, k), dtype=np.float64)
    for i in range(q):
        for j in range(k):
            omega[i, j] = gamma[i + j + 1]

    Gamma = np.zeros(shape=(k, k), dtype=np.float64)
    for i in range(k):
        for j in range(k):
            Gamma[i, j] = gamma[np.abs(i-j)]
    Gamma_inv = np.linalg.inv(Gamma)

    gamma_0 = gamma[0]
    gamma_1_q = np.array(gamma[1: q+1], dtype=np.float64)
    return omega, Gamma_inv, gamma_0, gamma_1_q


def calc_ma_phi_matrix(omega, gamma_0, gamma_1_q, Gamma_inv):
    """
    采用矩阵计算来估计 MA 模型的参数 phi. 估计的结果相当不准确. 但 sigma 估计得准确性还可以.
    :param omega: ndarray, 状为 (q, k), 每个位置 (i, j) 对应自相关函数 \gamma_{i+j+1}.
    :param gamma_0: \gamma_{0} 自相关函数.
    :param gamma_1_q: ndarray, 形状为 (q,)  \gamma_{1} 到 \gamma_{q} 的自相关函数组成.
    :param Gamma_inv: k 阶的自协方差矩阵的逆矩阵.
    :return: phi, sigma.
    phi: ndarray 形状为 (q+1,) .估计出的 phi 参数, \phi_{0} 到 \phi_{q}, 其中 \phi_{0} 一定为 1.
    sigma: 估计出的系统随机变量的标准差.
    """
    q = len(gamma_1_q)
    gamma_1_q = np.reshape(np.array(gamma_1_q, dtype=np.float64), newshape=(q, 1))

    a = np.eye(N=q, k=1)
    c = np.zeros(shape=(q, 1))
    c[0] = 1

    pi = np.dot(np.dot(omega, Gamma_inv), omega.T)

    sigma_square = np.squeeze(gamma_0 - np.dot(np.dot(c.T, pi), c))

    phi = (gamma_1_q - np.dot(np.dot(a, pi), c)) / sigma_square
    phi = np.squeeze(phi, axis=1)
    phi = np.insert(phi, 0, 1)
    sigma = np.sqrt(sigma_square)
    return phi, sigma


def demo1():
    """
    ACF 自相关函数.
    AR 模型的 ACF (AutoCovariance Function) 是拖尾的. 无法看出模型的阶数.
    """
    # 生成时间序列.
    phi_list = [0.7]
    # phi_list = [1, -0.8]

    generator = auto_regression_q(phi_list)
    ts = list()
    for i in range(3000):
        ts.append(next(generator))
    x = np.array(ts[100:-100], dtype=np.float64)

    # 计算自相关函数.
    q = 20
    gamma = acf(x, q)

    plt.bar(np.arange(q+1), gamma, width=0.5, bottom=None)
    plt.show()
    return


def demo2():
    """
    计算自 AR 回归模型的 phi 系数, 可以通过系数判断模型的阶数.
    """
    # 生成时间序列.
    phi_list = [0.7]
    # phi_list = [1, -0.8]

    generator = auto_regression_q(phi_list, mu=1, sigma=5)
    ts = list()
    for i in range(3000):
        ts.append(next(generator))
    x = np.array(ts[100:-100], dtype=np.float64)

    # 计算自相关函数.
    q = 10
    phi_predict, gamma, Gamma = pacf(x, q)
    print(f"predict phi: {phi_predict}")
    plt.bar(np.arange(q), phi_predict, width=0.5, bottom=None)
    plt.show()

    # 计算系统随机变量 z 的期望和方差.
    ez = (1 - np.sum(phi_predict)) * np.mean(x)
    gamma_0 = Gamma[0, 0]
    dz = np.sqrt(gamma_0 - np.dot(phi_predict, gamma))
    print(f"系统随机变量 z 的期望和方差: E(Z): {ez}, D(Z): {dz}")
    return


def demo3():
    """
    白噪声的 ACF 自相关函数
    """
    # 生成时间序列.
    generator = white_noise()
    ts = list()
    for i in range(3000):
        ts.append(next(generator))
    x = np.array(ts[100:-100], dtype=np.float64)

    # 计算自相关函数. q 猜想的模型的阶数.
    q = 20
    gamma = acf(x, q)
    print(f"白噪声的自相关函数计算: {gamma}")
    plt.bar(np.arange(q+1), gamma, width=0.5, bottom=None)
    plt.show()
    return


def demo4():
    """滑动平均模型, 通过自相关函数作 阶数估计"""
    phi_list = [0.36, 0.85]
    generator = moving_average_q(phi_list, mu=0, sigma=1)
    ts = list()
    for i in range(3000):
        ts.append(next(generator))
    x = np.array(ts[100:-100], dtype=np.float64)

    # 计算自相关函数. 用于估计阶数.
    q = 20
    # gamma = acf(x, q)
    gamma_matrix = get_gamma_matrix(x, q)
    gamma = gamma_matrix[0]
    print(f"MA 滑动平均模型, 通过自相关函数作阶数估计: {gamma}")
    plt.bar(np.arange(q+1), gamma, width=0.5, bottom=None)
    plt.show()
    return


def demo5():
    """对 MA 滑动平均模型计算 PACF 得出的序列是托尾的. """
    phi_list = [0.83, -0.12]
    generator = moving_average_q(phi_list, mu=1, sigma=5)
    ts = list()
    # 加大时间序列值的数量可以提高估计准确度.
    for i in range(100000):
        ts.append(next(generator))
    x = np.array(ts[100:-100], dtype=np.float64)
    # x = np.array(ts, dtype=np.float64)

    q = 20
    phi_predict, gamma, Gamma = pacf(x, q)
    print(f"predict phi: {phi_predict}")
    plt.bar(np.arange(q), phi_predict, width=0.5, bottom=None)
    plt.show()
    return


def demo6():
    """滑动平均模型, 线性迭代法参数估计."""
    phi_list = [0.36, 0.85]
    generator = moving_average_q(phi_list, mu=0, sigma=1)
    ts = list()
    for i in range(1000000):
        ts.append(next(generator))
    # x = np.array(ts[100:-100], dtype=np.float64)
    x = np.array(ts, dtype=np.float64)

    # 通过自相关函数可以估计出 MA 模型为2阶.
    # 求 gamma0 gamma1 gamma2
    q = 2
    gamma = acf(x, q)
    # plt.bar(np.arange(q+1), gamma, width=0.5, bottom=None)
    # plt.show()

    # 初始化参数 \phi.
    # phi_init = np.array([1, 0.36, 0.85], dtype=np.float64)
    phi_init = np.array([1, 0.99, 0.1], dtype=np.float64)
    # phi_init = np.array([1, gamma[1], gamma[2]], dtype=np.float64)

    phi_estimate_iter, loss = calc_ma_phi_iter(gamma, phi_init)
    print(f"phi_estimate_iter: {phi_estimate_iter}, loss: {loss}")
    return


def demo7():
    """采用矩阵计算来估计 MA 模型的参数 phi. """
    phi_list = [0.36, 0.85]
    generator = moving_average_q(phi_list, mu=0, sigma=1)
    ts = list()
    for i in range(100000):
        ts.append(next(generator))
    # x = np.array(ts[100:-100], dtype=np.float64)
    x = np.array(ts, dtype=np.float64)

    q = 2
    k = 100
    omega, Gamma_inv, gamma_0, gamma_1_q = calc_omega(x, q, k)
    phi_estimate_matrix, sigma = calc_ma_phi_matrix(omega, gamma_0, gamma_1_q, Gamma_inv)
    print(f"phi_estimate_matrix: {phi_estimate_matrix}")

    return


def demo8():
    """矩阵法估计和线性迭代法估计. """
    phi_list = [0.36, 0.85]
    generator = moving_average_q(phi_list, mu=1, sigma=5)
    ts = list()
    # 加大时间序列值的数量可以提高估计准确度.
    for i in range(1000000):
        ts.append(next(generator))
    x = np.array(ts[100:-100], dtype=np.float64)
    # x = np.array(ts, dtype=np.float64)

    # 矩阵法估计.
    q = 2
    k = 100
    omega, Gamma_inv, gamma_0, gamma_1_q = calc_omega(x, q, k)
    phi_estimate_matrix, sigma_estimate_matrix = calc_ma_phi_matrix(omega, gamma_0, gamma_1_q, Gamma_inv)
    print(f"phi_estimate_matrix: {phi_estimate_matrix}, sigma_estimate_matrix: {sigma_estimate_matrix}")

    # 线性迭代法估计.
    phi_init = np.array(phi_estimate_matrix, dtype=np.float64)
    gamma = acf(x, q)
    phi_estimate_iter, loss = calc_ma_phi_iter(gamma, phi_init)
    print(f"phi_estimate_iter: {phi_estimate_iter}, loss: {loss}")

    # 根据 phi 求系统随机变量的期望与方差.
    mu = np.mean(x) / np.sum(phi_estimate_iter)
    sigma = np.sqrt(gamma_0 / np.dot(phi_estimate_iter, phi_estimate_iter))
    print(f"系统随机变量的期望与方差: mu: {mu}, sigma: {sigma}")
    return


if __name__ == '__main__':
    # demo1()
    # demo2()
    # demo3()
    # demo4()
    # demo5()
    demo6()
    # demo7()
    # demo8()
