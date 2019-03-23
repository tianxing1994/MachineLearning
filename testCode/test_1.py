from sklearn.preprocessing import scale
from sklearn.datasets import load_iris

data = load_iris().data

data_scaled = scale(data,axis=1)
print(data_scaled)
