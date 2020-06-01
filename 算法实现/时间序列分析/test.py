import numpy as np


phi_list = [0.9, 0.7, 0.5, 0.3, 0.1]
# phi_list = [1, -0.8]

m_n = len(phi_list)
f = np.eye(m_n, m_n, -1, dtype=np.float64)
f[0] = np.array(phi_list, dtype=np.float64)
print(f)

eigvalue, eigvector = np.linalg.eig(f)
print(eigvalue)

p = len(eigvalue)
c_denominator = list()
for i in range(p):
    c_denominator_temp = 1
    for j in range(p):
        if i == j:
            continue
        c_denominator_temp *= (eigvalue[i] - eigvalue[j])
    c_denominator.append(c_denominator_temp)

c = list()
for i in range(p):
    c_temp = eigvalue[i]**(p-1) / c_denominator[i]
    c.append(c_temp)

c = np.array(c, dtype=np.complex)

for i in range(100):
    lambda_j = np.power(eigvalue, i+1)
    f11_j = np.dot(c, lambda_j)
    print(f11_j)
