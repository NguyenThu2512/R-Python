#%% - Nạp thư viện
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from pmdarima.arima import auto_arima
from statsmodels.datasets import sunspots
#%% - some config
plt.rcParams['figure.figsize']=(10,8)
plt.rcParams['figure.dpi']=200
plt.rcParams['font.size']=13
#%%- nạp dữ liệu
df=pd.read_csv("./data/Buys_computer.csv")
x=df.drop("Buys_computer", axis='columns')
y=df['Buys_computer']
df.info()

#%%
from sklearn.preprocessing import LabelEncoder
x['Age_n']=LabelEncoder().fit_transform(x['Age'])
# d={'<=30':0, '31..40': 1, '>40':2}
# x['Age_n']=x['Age'].map(d)
x['Income_n']=LabelEncoder().fit_transform(x['Income'])
x['Student_n']=LabelEncoder().fit_transform(x['Student'])
x['Credit_rating_n']=LabelEncoder().fit_transform(x['Credit_rating'])
x_n=x.drop(['Age', 'Income','Student','Credit_rating'],axis='columns')
y_n=LabelEncoder().fit_transform(y)

#%%- fit model
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
# support criteria are "gini" for the Gini index and "entropy" for the information gain
model=DecisionTreeClassifier(criterion='entropy',random_state=10).fit(x_n,y_n)
# model=DecisionTreeClassifier(criterion='entropy', random_state=100).fit(x_n, y_n)
#Trước phần này phải chia bộ dữ liệu train/test
#%%
score=model.score(x_n, y_n)
#Dộ chính xác của mô hình(100%) => k nên làm như vậy

#%%- visualize result
features=['Age', 'Income', 'Student', 'Credit_rating']
text_representation=tree.export_text(model,feature_names=features)
print(text_representation)
#%%
plt.figure(figsize=(20,20), dpi=150)
t=tree.plot_tree(model, feature_names=features, class_names=['No', 'Yes'], filled=True)
plt.show()

#%%- Prediction
#Age 30, Income: low, Student: no, credit:fair
buy_computer=model.predict([[1,1,0,1]])
print(buy_computer)


