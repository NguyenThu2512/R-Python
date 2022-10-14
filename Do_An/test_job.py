#%%- import library
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#for preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling  import SMOTE

#for model score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score , recall_score , f1_score

#for models
import plotly.express as px
from sklearn import metrics
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import plot_roc_curve, roc_auc_score, roc_curve

#%%nạp dữ liệu
df=pd.read_csv("./Data/data_job.xlsx")
print(df.head())
print(df.shape)
print(df.info())