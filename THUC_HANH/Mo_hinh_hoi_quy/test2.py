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
f= open("./data/Advertising-Budget-and-Sales-_1_.txt")
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
tv=[]
radio=[]
new=[]
sales=[]
end=len(my_list_1)
for i in range(4,end,5):
    my_list_1[i]=int(my_list_1[i])
    stt.append(my_list_1[i])
print(stt)
for i in range(5,end,5):
    my_list_1[i]=float(my_list_1[i])
    tv.append(my_list_1[i])
print(tv)
for i in range(6,end,5):
    my_list_1[i] = float(my_list_1[i])
    radio.append(my_list_1[i])
print(radio)
for i in range(7,end,5):
    my_list_1[i] = float(my_list_1[i])
    new.append(my_list_1[i])
print(new)
for i in range(8,end,5):
    my_list_1[i] = float(my_list_1[i])
    sales.append(my_list_1[i])
print(sales)
df=pd.DataFrame(data=tv)
radio=pd.DataFrame(data=radio)
new=pd.DataFrame(data=new)
sales=pd.DataFrame(data=sales)
df = df.assign(radio=radio.values)
df = df.assign(new=new.values)
df = df.assign(sales=sales.values)
# cost_ad=[tv,radio,new,sales]
# df=pd.DataFrame(data=cost_ad)
df.columns=['tv','radio','new','sales']
print(len(stt))
print(len(tv))
print(len(radio))
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
#%%-remove outliner
df1 = df.copy()
df3 = df1.copy()
df1 = df3.copy()
##
#%%-mô tả thống
df.describe()
# std = StandardScaler()
# df_std = std.fit_transform(df)
# df_std = pd.DataFrame(df)
# display(df_std.describe())
#%% - Chia giá trị dữ liệu của 4 biến theo 3 mức Low, Medium và High dựa trên các trị thống kê
# t = ['Low','Medium','High']
# tv_ = [0, 74.375, 218.825, 296.400] #Các giá trị lần lượt là[min, 25%, 75%, max]
# df['TV_']= pd.cut(df['TV'], bins=tv_,labels=t)
# radio_ = [0, 9.975, 36.525, 49.600]
# df['Radio_']= pd.cut(df['Radio'], bins=radio_,labels=t)
# newspaper_ = [0, 12.75, 45.10, 114]
# df['Newspaper_']= pd.cut(df['Newspaper'],
# bins=newspaper_,labels=t)
# sales_ = [0, 10, 20, 30]
# df['Sales_']= pd.cut(df['Sales'], bins=sales_,labels=t)
from sklearn.preprocessing import LabelEncoder
x['Age_n']=LabelEncoder().fit_transform(x['Age'])
# d={'<=30':0, '31..40': 1, '>40':2}
# x['Age_n']=x['Age'].map(d)
x['Income_n']=LabelEncoder().fit_transform(x['Income'])
x['Student_n']=LabelEncoder().fit_transform(x['Student'])
x['Credit_rating_n']=LabelEncoder().fit_transform(x['Credit_rating'])
x_n=x.drop(['Age', 'Income','Student','Credit_rating'],axis='columns')
y_n=LabelEncoder().fit_transform(y)






