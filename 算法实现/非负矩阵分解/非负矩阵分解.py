"""
参考链接:
https://www.cnblogs.com/gavanwanggw/p/7337227.html
"""
import numpy as np


def nmf(v, n_topics, max_iters=100, epsilon=1e-5):
    k = n_topics
    m, n = np.shape(v)
    w = np.array(np.random.random((m, k)))
    h = np.array(np.random.random((k, n)))
    for i in range(max_iters):
        v_pred = np.dot(w, h)
        loss = np.sum(np.power(v_pred - v, 2))
        # print(f"iter: {i}, loss: {loss}")
        if loss < epsilon:
            break
        # 乘性更新
        matrix_h = np.dot(w.T, v) / np.dot(np.dot(w.T, w), h)
        h = h * matrix_h
        matrix_w = np.dot(v, h.T) / np.dot(np.dot(w, h), h.T)
        w = w * matrix_w
    return w, h


if __name__ == '__main__':
    w0 = np.array([[1, 0],
                   [0, 1]])
    h0 = np.array([[0.3, 0.4, 0.5, 0.6, 0.7, 0.7, 0.1, 0.1, 0.2, 0.1],
                   [0.4, 0.3, 0.2, 0.1, 0.1, 0.2, 0.5, 0.6, 0.2, 0.8]])
    v0 = np.dot(w0, h0)
    # print(v0)
    w1, h1 = nmf(v0, 2, max_iters=1000, epsilon=1e-9)
    v1 = np.dot(w1, h1)
    print(w1)
    # print(h1)
    # print(v1)
