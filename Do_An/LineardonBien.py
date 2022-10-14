# %% Import library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import statsmodels.api as sms
#%% Some configs
plt.rcParams['figure.figsize'] = (10,8)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 15
# %% Load data
df = pd.read_csv('./data/Car_sales_final.csv')
df.head()
# %% Kiểm tra dữ liệu Null
df.isnull().sum()
# %% Điền vào các dữ liệu Null sau khi dùng phép loại suy
df = df.interpolate()
print(df)
#%% Dữ liệu sau khi làm sạch
df.isnull().sum()

# %% Xem chi tiết tập dữ liệu
print(df.describe())
#%% Khai báo biến
price = df[["Price_in_thousands"]].values.reshape(-1,1)
sale = df['Sales_in_thousands'].values.reshape(-1,1)
resale = df['__year_resale_value'].values.reshape(-1,1)
enginesize = df['Engine_size'].values.reshape(-1,1)
horsepower = df['Horsepower'].values.reshape(-1,1)
wheelbase = df['Wheelbase'].values.reshape(-1,1)
width = df['Width'].values.reshape(-1,1)
length = df['Length'].values.reshape(-1,1)
curbweight = df['Curb_weight'].values.reshape(-1,1)
fuelcapacity = df['Fuel_capacity'].values.reshape(-1,1)
fuelefficiency = df['Fuel_efficiency'].values.reshape(-1,1)
power = df['Power_perf_factor'].values.reshape(-1,1)


# %% Vẽ biểu đồ Scatter phân tán giữa các biến
fig, ((ax1, ax2, ax3, ax4, ax5, ax6), (ax7, ax8, ax9, ax10, ax11, ax12)) = plt.subplots(ncols = 6, nrows = 2, figsize=(50,20))
ax1.scatter(sale, price)
ax1.set(xlabel = 'Sale ')
ax1.set(ylabel= 'Prices in thousands')
ax1.set(title = 'Scatter giữa Giá và Sales_in_thousands')

ax2.scatter(resale, price)
ax2.set(xlabel = 'Resale')
ax2.set(ylabel= 'Prices in thousands')
ax2.set(title = 'Scatter giữa Giá và__year_resale_value')

ax3.scatter(enginesize, price)
ax3.set(xlabel = 'Enginesize')
ax3.set(ylabel= 'Prices in thousands')
ax3.set(title = 'Scatter giữa Giá và Engine_size')

ax4.scatter(horsepower, price)
ax4.set(xlabel = 'Horsepower')
ax4.set(ylabel= 'Prices in thousands')
ax4.set(title = 'Scatter giữa Giá và Horsepower')

ax5.scatter(wheelbase, price)
ax5.set(xlabel = 'Wheelbase')
ax5.set(ylabel= 'Prices in thousands')
ax5.set(title = 'Scatter giữa Giá và Wheelbase')

ax6.scatter(width, price)
ax6.set(xlabel = 'Width')
ax6.set(ylabel= 'Prices in thousands')
ax6.set(title = 'Scatter giữa Giá và Width')

ax7.scatter(length , price)
ax7.set(xlabel = 'Length')
ax7.set(ylabel= 'Prices in thousands')
ax7.set(title = 'Scatter giữa Giá và Length')

ax8.scatter(curbweight, price)
ax8.set(xlabel = 'Curbweight')
ax8.set(ylabel='Prices in thousands')
ax8.set(title = 'Scatter giữa Giá và Curb_weight')

ax9.scatter(fuelcapacity, price)
ax9.set(xlabel = 'Fuelcapacity')
ax9.set(ylabel= 'Prices in thousands')
ax9.set(title = 'Scatter giữa Giá và Fuel_capacity')

ax10.scatter(fuelefficiency, price)
ax10.set(xlabel = 'Fuelefficiency')
ax10.set(ylabel= 'Prices in thousands')
ax10.set(title = 'Scatter giữa Giá và Fuel_efficiency')

ax11.scatter(power, price)
ax11.set(xlabel = 'Power')
ax11.set(ylabel= 'Prices in thousands')
ax11.set(title = 'Scatter giữa Giá và Power_perf_factor')

plt.show()

#Nhìn vào biểu đồ ta có các biến có mối tương quan với Price: Sale_in_thousands, __year_sale_value, Engine size,
# Horse power, Curb_weight, Fuel_capacity, Power_perf_factor, Power_perf_factor

#%% Vẽ heatmap để xác định tương quan mạnh yếu giữa các biến
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 250
plt.rcParams['figure.figsize'] = (13,5)
sns.heatmap(df.corr(), annot =True, linewidths=.5)
plt.show()

#%% Create model
model = LinearRegression().fit(power, price)

#%% Get results
intercept = model.intercept_
slope = model.coef_
R_square = model.score(power, price)
print("Hệ số R_square:",R_square)
print("Hệ số chặn:",intercept)
print("Hệ số góc:",slope)

# R_square ax11 là tốt nhất 80%
#%% - Create model by OLS
# power = sms.add_constant(power)
# model = sms.OLS(price, power)
# results = model.fit()
# print(results.summary())
#%% Prediction
predictive_values = model.predict(power)
#%%
plt.scatter(power, price)
plt.plot(power, predictive_values, color='r')
# plt.set(xlabel = 'Công suất cực đại')
# plt.set(ylabel= 'Giá xe')
# plt.set(title = 'Scatter giữa Giá và Power_perf_factor')
plt.show()
#%% Future Prediction
future_values = np.array([200, 150, 300]).reshape(-1, 1)
# predicted_values = intercept * slope * future_values
f_predicted_values = model.predict(future_values)
print(f_predicted_values)