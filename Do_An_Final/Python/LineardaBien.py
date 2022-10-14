# %% Import library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import statsmodels.api as sms
from test import *
from statsmodels.stats.outliers_influence import variance_inflation_factor
#%% Some configs
plt.rcParams['figure.figsize'] = (10,8)
plt.rcParams['figure.dpi'] = 50
plt.rcParams['font.size'] = 15

# %% Load data
df1 = pd.read_csv('./data/Car_sales_final.csv')
df1.head()
# %% Làm sạchh
df1.isnull().sum()
# %%
df1 = df1.interpolate()
print(df1)
#%%
df1.isnull().sum()
# %%
df = df1.copy()
# %% Create model
x_list_new = df[['Sales_in_thousands','__year_resale_value','Engine_size','Horsepower','Wheelbase','Width','Length','Curb_weight','Fuel_capacity','Fuel_efficiency','Power_perf_factor']]
x = x_list_new
y = df[["Price_in_thousands"]]
x = sms.add_constant(x)
model = sms.OLS(y, x)
results = model.fit()
print(results.summary())

# %% Split train/test dataset chia tập dữ liệu
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.1)
# %% Model 1
x_list_new = x_list_new.drop(["__year_resale_value"], axis = 1)
x1 = x_list_new
y1 = df[["Price_in_thousands"]]
x1 = sms.add_constant(x1)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x1,y1, test_size=0.1)
new_model = sms.OLS(y_train, x_train)
new_results = new_model.fit()
print(new_results.summary())

# %% Model 2
x_list_new = x_list_new.drop(["Fuel_capacity"], axis = 1)
x1 = x_list_new
y1 = df[["Price_in_thousands"]]
# x1, y1 = np.array(x1), np.array(y1)
x1 = sms.add_constant(x1)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x1,y1, test_size=0.1)
new_model = sms.OLS(y_train, x_train)
new_results = new_model.fit()
print(new_results.summary())
# %% Model 3
x_list_new = x_list_new.drop(["Sales_in_thousands"], axis = 1)
x1 = x_list_new
y1 = df[["Price_in_thousands"]]
# x1, y1 = np.array(x1), np.array(y1)
x1 = sms.add_constant(x1)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x1,y1, test_size=0.1)
new_model = sms.OLS(y_train, x_train)
new_results = new_model.fit()
print(new_results.summary())
# %% Model 4
x_list_new = x_list_new.drop(["Length"], axis = 1)
x1 = x_list_new
y1 = df[["Price_in_thousands"]]
# x1, y1 = np.array(x1), np.array(y1)
x1 = sms.add_constant(x1)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x1,y1, test_size=0.1)
new_model = sms.OLS(y_train, x_train)
new_results = new_model.fit()
print(new_results.summary())
# %% Model 5
x_list_new = x_list_new.drop(["Wheelbase"], axis = 1)
x1 = x_list_new
y1 = df[["Price_in_thousands"]]
# x1, y1 = np.array(x1), np.array(y1)
x1 = sms.add_constant(x1)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x1,y1, test_size=0.1)
new_model = sms.OLS(y_train, x_train)
new_results = new_model.fit()
print(new_results.summary())
# %% Model 6
x_list_new = x_list_new.drop(["Width"], axis = 1)
x1 = x_list_new
y1 = df[["Price_in_thousands"]]
# x1, y1 = np.array(x1), np.array(y1)
x1 = sms.add_constant(x1)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x1,y1, test_size=0.1)
new_model = sms.OLS(y_train, x_train)
new_results = new_model.fit()
print(new_results.summary())
# %% Model 7
x_list_new = x_list_new.drop(["Curb_weight"], axis = 1)
x1 = x_list_new
y1 = df[["Price_in_thousands"]]
# x1, y1 = np.array(x1), np.array(y1)
x1 = sms.add_constant(x1)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x1,y1, test_size=0.1)
new_model = sms.OLS(y_train, x_train)
new_results = new_model.fit()
print(new_results.summary())
#%% Model 8
x_list_new = x_list_new.drop(["Fuel_efficiency"], axis = 1)
# x1 = x_list_new
x1  = x_list_new
y1 = df[["Price_in_thousands"]]
# x1, y1 = np.array(x1), np.array(y1)
x1 = sms.add_constant(x1)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x1,y1, test_size=0.1)
new_model = sms.OLS(y_train, x_train)
new_results = new_model.fit()
print(new_results.summary())
#%% Check VIF
vif_data = pd.DataFrame()
vif_data["feature"] = x1.columns
vif_data["VIF"] = [variance_inflation_factor(x1.values,i) for i in range (len(x1.columns))]
print(vif_data)
#Nếu hệ số phóng đại phương sai VIF (variance inflation factor) > 2 thì có dấu hiệu đa cộng tuyến,
# đây là điều không mong muốn.
# Nếu VIF > 10 thì chắc chắn có đa cộng tuyến. Nếu VIF <2: không bị đa cộng tuyến
# %%
x_list_new = x_list_new.drop(["Horsepower"], axis = 1)
# x1 = x_list_new
x1  = x_list_new
y1 = df[["Price_in_thousands"]]
# x1, y1 = np.array(x1), np.array(y1)
x1 = sms.add_constant(x1)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x1,y1, test_size=0.1)
new_model = sms.OLS(y_train, x_train)
new_results = new_model.fit()
print(new_results.summary())
#%% Check VIF
vif_data = pd.DataFrame()
vif_data["feature"] = x1.columns
vif_data["VIF"] = [variance_inflation_factor(x1.values,i) for i in range (len(x1.columns))]
print(vif_data)
#%%
x1  = df[['Horsepower', 'Engine_size']]
y1 = df[["Price_in_thousands"]]
# x1, y1 = np.array(x1), np.array(y1)
x1 = sms.add_constant(x1)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x1,y1, test_size=0.1)
new_model = sms.OLS(y_train, x_train)
new_results = new_model.fit()
print(new_results.summary())
#%% Check VIF
vif_data = pd.DataFrame()
vif_data["feature"] = x1.columns
vif_data["VIF"] = [variance_inflation_factor(x1.values,i) for i in range (len(x1.columns))]
print(vif_data)


#==> chọn mô hình gồm Engine_size, power_pert_factor
#%% Kiểm định mô hình với tập Test
x1  = df[['Engine_size', 'Power_perf_factor']]
y1 = df[["Price_in_thousands"]]
x1 = sms.add_constant(x1)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x1,y1, test_size=0.1)
new_model = sms.OLS(y_test, x_test)
new_results = new_model.fit()
print(new_results.summary())
#%% - Get results
R_square = new_results.rsquared
params = new_results.params
#%% - Predict
pre_values = params[0] + params[1] * df['Engine_size'] + params[2] * df['Power_perf_factor'] #không nhân biến income thì nó sẽ nhân theo 1 chuỗi gì đó, nói chung alf ko đc
#%% - Visualization
plt.plot(y,label = 'Actual')
plt.plot(pre_values, label = 'Predict', marker = 'x', color = 'red', linestyle = '--')
plt.legend()
plt.show()
#%%
other_x = np.array([[4.2, 200], [4.8, 150]])
other_x = sms.add_constant(other_x)
pre_future_values = new_results.predict(other_x)
print(pre_future_values)

