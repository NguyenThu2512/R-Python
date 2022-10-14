#%%-import library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sms

#%%- Load data
df=pd.read_csv("data/data_job.csv")
print(df.head())
print(df.shape)
print(df.info())
#%% - Trực quan hóa các biến liên tục

sns.boxplot(x = 'Exited', y = 'EstimatedSalary', data = df)
plt.tight_layout()
