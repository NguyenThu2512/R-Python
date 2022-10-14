#%%-import library
import time
import warnings
from sklearn.preprocessing import StandardScaler
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
from IPython.display import display

warnings.filterwarnings('ignore')
#%% - read data
my_list_1=[]
f= open("./Data/Ads.txt")
for line in f:
     lines = line.split("\n")
     n=lines[0].split(" ")
     print(n)
     for i in n:
         if i !='':
             my_list_1.append(i)
         else:
             continue
f.close()
print(my_list_1)
stt=[]
yt=[]
fb=[]
new=[]
sales=[]
end=len(my_list_1)
for i in range(4,end,5):
    my_list_1[i]=int(my_list_1[i])
    stt.append(my_list_1[i])
print(stt)
for i in range(5,end,5):
    my_list_1[i]=float(my_list_1[i])
    yt.append(my_list_1[i])
print(yt)
for i in range(6,end,5):
    my_list_1[i] = float(my_list_1[i])
    fb.append(my_list_1[i])
print(fb)
for i in range(7,end,5):
    my_list_1[i] = float(my_list_1[i])
    new.append(my_list_1[i])
print(new)
for i in range(8,end,5):
    my_list_1[i] = float(my_list_1[i])
    sales.append(my_list_1[i])
print(sales)
df=pd.DataFrame(data=yt)
radio=pd.DataFrame(data=fb)
new=pd.DataFrame(data=new)
sales=pd.DataFrame(data=sales)
df = df.assign(radio=radio.values)
df = df.assign(new=new.values)
df = df.assign(sales=sales.values)

df.columns=['Youtube','Facebook','Newspaper','Sales']
print(len(stt))
print(len(yt))
print(len(fb))
print(len(new))
print(len(sales))
df.info()
#%%- boxplot
plt.boxplot(df)
plt.show()
#%%-pairplot
sns.pairplot(df)
plt.show()
#%%-heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(df.corr(), annot=True)
plt.show()




