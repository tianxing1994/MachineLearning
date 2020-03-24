"""
通过线性回归将 x1,x2...xn, n 个变量的矩阵运算转化为 y 的单一变量,
根据 y 值求出事件 1 发生的概率函数 model(X,theta).
事件为 1 的概率函数为: model(X,theta)
事件为 0 的概率函数为: 1 - model(X,theta)
事件只能为 1 或 0, 则综合函数为: P(X, y, theta) = np.power(model(X,theta),y) * np.power((1 - model(X,theta)),(1-y))
如既有 m (m = len(X)) 个样本为 a 个 1, b 个 0, 1 发生的概率为 p, 则我们认为既有样本发生的概率为:  p^a * (1-p)^b.
则已知 a, b. 求 theta, 使得既有样本发生的概率最大.
似然函数: L(theta) = np.prod(P(X, y, theta))
我们需要求似然函数最大值处的 theta. 似然函数取对数, 将累乘变成累加.
对数似然函数: l(theta) = np.sum(y * np.log(model(X,theta)) + (1-y) * np.log((1 - model(X,theta))))
代价函数 cost_func(X, y, theta) = J(theta) = -1/m * l(theta)
需要求代价函数(损失函数)的最小值, 对代价函数求导得到:
gradient(X, y, theta) = △J(theta) / △theta = \
np.sum(np.multiply(-y, np.log(model(X, theta))) - np.multiply(1 - y, np.log(1 - model(X, theta)))) / (len(X))

梯度下降法, 即根据导数的值调整 theta 逼近代价函数取最小值时的 theta,
theta := theta + alpha * cost_func(X, y, theta)
theta 的取值点条件可以有三种方式"
1. 迭代固定次数, 取 theta
2. 当最近两次 theta 值对应的代价函数差值小于一定值时 △cost_func(X, y, theta) < threshold, 取 theta
3. 当 theta 值的调整幅度变得足够小时, gradient(X, y, theta) < threshold, 取 theta
"""
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing


def sigmoid(z):
    result = 1 / (1 + np.exp(-z))
    return result


def model(X, theta):
    result = sigmoid(np.dot(X, theta.T).reshape(-1, 1))
    return result


def cost_func(X, y, theta):
    """似然函数变换得到的代价函数."""
    left = np.multiply(-y, np.log(model(X, theta)))
    right = np.multiply(1 - y, np.log(1 - model(X, theta)))
    result = np.sum(left - right) / (len(X))
    return result


def gradient(X, y, theta):
    """似然函数变换得到的代价函数关于 theta 的导数 (cost 函数的导数)"""
    result = np.sum((model(X,theta) - y) * X, axis=0)
    return result


def shuffle_data(np_data):
    """打乱数据"""
    np.random.shuffle(np_data)
    X = np_data[:, :-1]
    y = np_data[:, -1].reshape(-1,1)
    return X, y


STOP_ITER = 0
STOP_COST = 1
STOP_GRAD = 2


def stop_criterion(value,stop_type=0, threshold=100.):
    """设定三种不同的停止策略"""
    if stop_type == STOP_ITER:
        return value >= threshold
    elif stop_type == STOP_COST:
        return abs(value[-1]-value[-2]) < threshold
    elif stop_type == STOP_GRAD:
        return np.linalg.norm(value) < threshold


def descent(np_data, theta, batch_size, alpha=0.000001, stop_type=0, threshold=5000.):
    """
    梯度下降求解逻辑回归, 得出最佳 theta 值
    推荐参数: stop_type=0, alpha=0.000001, thresh=5000. stop_type=1, alpha=0.001, thresh=0.000001.
    """
    init_time = time.time()

    l = len(np_data)
    if batch_size > l: batchSize = int(l/10)

    i = 0   # 迭代次数
    k = 0   # 小批量数据起点

    X, y = shuffle_data(np_data)
    cost = cost_func(X, y, theta)
    costs = [cost,]

    status = 1  # 如果函数正常退出, 值为 0.

    while True:
        grad = gradient(X[k:k + batch_size], y[k:k + batch_size], theta)
        theta -= grad * alpha   # 代价函数求最小值, grad 为正则, theta 应减去它.
        k += batch_size
        if k + batch_size > l: # 数据已取完, 重新洗牌, 继续.
            k = 0
            X, y = shuffle_data(np_data)

        costs.append(cost_func(X, y, theta))

        if stop_type == STOP_ITER:
            value = i
        elif stop_type == STOP_COST:
            value = costs
        elif stop_type == STOP_GRAD:
            value = grad
        else:
            value = i

        if stop_criterion(value,stop_type,threshold):
            status = 0
            time_cost = time.time()-init_time
            break

        i += 1
    # 好的退出时机应是 grad 值接近于 0.
    return grad, theta, i, costs, status, time_cost, alpha


def plotmat(grad, theta, i, costs, status, time_cost, alpha):
    """根据 descent 函数的返回值画出 cost 与迭代次数的变化趋势."""
    last_cost = costs[-1]
    text = "last_cost: " + "%0.3f" % last_cost + ", time_cost: " + "%0.4f" % last_cost + "\n" + "theta: " + str(theta)

    plt.plot(np.arange(len(costs)), costs, "r")
    plt.title(text)
    plt.show()
    return


def predict(X, theta):
    """根据 X, theta 预测结果的函数"""
    result = [1 if x >= 0.5 else 0 for x in model(X, theta)]
    return result


def score(X, y, theta):
    """根据 X, y, theta 计算得分的函数"""
    predictions = [1 if x >= 0.5 else 0 for x in model(X, theta)]
    correct = [1 if (a == b) else 0 for (a, b) in zip(predictions, y)]
    result = sum(correct) / len(correct)
    return result


def load_data(path):
    """读取与创建基本数据"""
    pd_data = pd.read_csv(path, header=None, names=['Exam 1', 'Exam 2', 'Admitted'])
    pd_data.insert(0, "Ones", 1)
    np_data = pd_data.values
    np_data_scaled = np_data.copy()
    np_data_scaled[:, 1:-1] = preprocessing.scale(np_data[:, 1:-1])

    X = pd_data.iloc[:, :-1].values
    y = pd_data.iloc[:, -1].values.reshape(-1, 1)

    X_scaled = X.copy()
    X_scaled[:, 1:] = preprocessing.scale(X[:, 1:])
    return np_data, np_data_scaled, X_scaled, y


def demo1():
    np_data, np_data_scaled, X_scaled, y = load_data("../dataset/classify_data/2d_binary_classification.txt")
    theta = np.array([0, 0, 0], dtype=np.float64)

    # grad, theta, i, costs, status, time_cost, alpha = descent(np_data, theta, batch_size=10, alpha=0.000001, stop_type=0, threshold=5000)
    # grad, theta, i, costs, status, time_cost, alpha = descent(np_data, theta, batch_size=1, alpha=0.000001, stop_type=1, threshold=0.000001)
    # grad, theta, i, costs, status, time_cost, alpha = descent(np_data, theta, batch_size=1, alpha=0.0001, stop_type=2, threshold=0.05)
    grad, theta, i, costs, status, time_cost, alpha = descent(np_data_scaled, theta, batch_size=100, alpha=0.0001, stop_type=0, threshold=50000)

    # print("结果模型的预测结果: ", predict(X_scaled, theta))
    print("结果模型在训练集上的得分: ", score(X_scaled, y, theta))
    plotmat(grad, theta, i, costs, status, time_cost, alpha)
    return


if __name__ == '__main__':
    demo1()
