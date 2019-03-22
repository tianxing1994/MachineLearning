import numpy as np
import pandas as pd

test_labels = pd.DataFrame([1,0,0,1,1])
predict_labels = pd.DataFrame([0,0,1,1,1])

target = np.logical_xor(test_labels>0,predict_labels>0)
print(target)