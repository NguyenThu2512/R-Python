
#%% - Import library
import numpy as np
arr=np.linspace(5,15,6,endpoint=False)
print(arr)
arr1=np.random.randint(1,45,1)
print(arr1)
#%% - Import library
import pandas as pd

#%% - Load data
df = pd.read_csv('data/TCB_2018_2020.csv', index_col=0)
# print(df[[0,2,3]].head())
print(df.loc['2020-06-15'])
#%% - Import library
import matplotlib.pyplot as plt

#%% - some config
plt.rcParams['figure.figsize']=(10,8)
plt.rcParams['figure.dpi']=200
plt.rcParams['font.size']=13
plt.rcParams['savefig.dpi']=200
plt.rcParams['legend.fontsize']='large'
plt.rcParams['figure.titlesize']='medium'

#%% - Load data
df=pd.read_csv('data/NetProfit.csv')
dat=df[['Year','VIC']]


#%%-Visualization
plt.plot('Year','VIC',data=dat)
plt.show()

