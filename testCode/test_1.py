import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import auc, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from scipy import interp
import pickle


result_df = pd.DataFrame(data=None, columns=["threshold","total_correct","Recall"])
result_df = result_df.append([[0.1, "5", 4]],ignore_index=True)

print(result_df)