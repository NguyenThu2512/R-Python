#%%-import library
import time
import warnings
import numpy as np
import json
import seaborn as sns
from pandas.io.json import json_normalize
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sms
import ijson
from pandas.io.json import json_normalize
from seaborn import heatmap
from sklearn.linear_model import LinearRegression
warnings.filterwarnings('ignore')
import xml.etree.ElementTree as ET
import pandas as pd
#%%- load data
with open("./Data/Prices_of_Cars.json") as f:
    d=json.load(f)
data_df = pd.io.json.json_normalize(d)
data_df.columns=['Hang','Km','Nam','Gia']
data_df['Gia'] = data_df['Gia'].astype(float)
data_df['Km'] = data_df['Km'].astype(float)
data_df['Nam'] = data_df['Nam'].astype(int)
#%%- tương quan giữa giá xe và số năm sử dụng
plt.scatter(data_df['Nam'],data_df['Gia'])
plt.xlabel("Năm sử dụng")
plt.ylabel("Giá xe")
plt.show()
#%%- tương quan giữa giá xe và số km đã đi
plt.scatter(data_df['Km'],data_df['Gia'])
plt.xlabel("Số km ")
plt.ylabel("Giá xe")
plt.show()
#%% - heatmap
sns.heatmap(data_df[["Gia","Km","Nam"]].corr(), annot=True,linewidths=.5,fmt='.1f')
plt.title("Correlation Graph",c="r",size=25)
plt.show()

#%%-Create model
x=data_df[["Km","Nam"]]
y=data_df[['Gia']]
x1=sms.add_constant(x)
# model=LinearRegression().fit
model=sms.OLS(y,x1).fit()
print(model.summary())
#%%- Predict
other_x=np.array([90,6],[15.5,3])
other_x=sms.add_constant(other_x)
pre_future_values=model.predict(other_x)
#%% Save model
import pickle
with open('./Model/model_pickle','wb') as f:
    pickle.dump(model, f)


with open('./Model/model_pickle','rb') as f:
    model=pickle.load(f)

# pred_value=model.predict(future_value)