import numpy as np

p = 5
phi_list = [0.9, 0.7, 0.5, 0.3, 0.1]

f = np.array([[0.9, 0.7, 0.5, 0.3, 0.1],
              [1, 0, 0, 0, 0],
              [0, 1, 0, 0, 0],
              [0, 0, 1, 0, 0],
              [0, 0, 0, 1, 0]], dtype=np.float64)


def f_j(f_matrix, j):
    ret = f_matrix
    for i in range(j):
        ret = np.dot(ret, ret)
    return ret


for i in range(100):
    ret = f_j(f, i)
    print(ret[0, 0])
