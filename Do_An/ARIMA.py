#%% - Nạp thư viện
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from pmdarima.arima import auto_arima
import warnings
warnings.filterwarnings("ignore")
#%% - Thiết lập thông số cơ bản cho biểu đồ
plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 12

#%% - Nạp dữ liệu phân tích
df = pd.read_csv("./data/Portland-oregon-average-monthly.csv", index_col='Month', parse_dates=True)
df.info()

#%% - Vẽ biểu đồ
plt.plot(df['traffic'])
plt.xlabel("Date")
plt.ylabel("Traffic")
plt.show()
#%% - Chia tập dữ liệu huấn luyện (train: 9%) và kiểm thử (test: 10%)
df_traffic = np.log(df['traffic'])
train_data, test_data = df_traffic[:int(len(df_traffic)*0.9)], df_traffic[int(len(df_traffic)*0.9):]
plt.plot(train_data, 'blue', label = 'Train data')
plt.plot(test_data, 'red', label = 'Test data')
plt.xlabel('Date')
plt.ylabel('Traffic')
plt.legend()
plt.show()

#%% - Biểu đồ lịch sử so sánh lượng hành khách với giá trị trung bình và độ lệch chuẩn của 12 kỳ trước đó
rolmean = train_data.rolling(12).mean()
rolstd = train_data.rolling(12).std()
plt.plot(train_data, color = 'blue', label ="Original")
plt.plot(rolmean, color = 'red', label = "Rolling mean")
plt.plot(rolstd, color = 'green', label = 'Rolling std' )
plt.legend()
plt.show()

#%% - Biểu đồ phân rã chuỗi thời gian (decompose)
decompose_results = seasonal_decompose(train_data, model = 'multiplicative', period = 4)
decompose_results.plot()
plt.show()

#%% Kiểm định tính đúng của dữ liệu (stationary)
def adf_test(data):
    indices = ["ADF: Test statistic", "p value","H of lags", "H of observatión"]
    test = adfuller(train_data, autolag = "AIC")
    results = pd.Series(test[:4], index=indices)
    for key, value in test[4].items():
        results[f"Critical Value ({key})"] = value
    return results

def kpss_test(data):
    indices = ["KPSS: Test statistic","p value","H of lags" ]
    test = kpss(data)
    results = pd.Series(test[:3], index=indices)
    for key, value in test[3].items():
        results[f"Critical Value ({key})"] = value
    return results

print(adf_test(train_data))
print("----"*5)
print(kpss_test(train_data))

#%% - Kiểm định tự tương quan
pd.plotting.lag_plot(train_data)
plt.show()
#%%
plot_pacf(train_data)
plt.show()
#%%
plot_acf(train_data)
plt.show()
#%% Chuyển đổi chuỗi dừng
#Tính sai phân bậc 1 dữ liệu train
diff = train_data.diff(1).dropna()
#Biểu đồ thể hiện dữ liệu ban đầu và sau khi lấy sai phần
fig, ax = plt.subplots(2, sharex = "all")
train_data.plot(ax=ax[0], title = "Traffic")
diff.plot(ax=ax[1], title ="Sai phân bậc nhất")
plt.show()

#%% - Kiểm tra lại tính dừng của dữ liệu sau khi lấy sai phần
print(adf_test(diff))
print("-------"*5)
print(kpss_test(diff))
plot_pacf(diff)
plt.show()

#%%
plot_acf(diff)
plt.show()
#%% xác định tham số "p, d, q" cho mô hình ARIMA
stepwise_fit = auto_arima(train_data, trace = True,suppress_warnings = True )
print(stepwise_fit.summary())
stepwise_fit.plot_diagnostics(figsize=(15, 8))
plt.show()

#%% Tạo model

model = sm.tsa.arima.ARIMA(train_data, order=(0, 1, 0))
fitted = model.fit()
print(fitted.summary())

#%% Vẽ biểu đồ phần dư (Residual Plot)
residuals = pd.DataFrame(fitted.resid)
fig, ax = plt.subplots(1,2)
residuals.plot(title ='Residual', ax = ax[0])
residuals.plot(title='Density',kind='kde', ax=ax[1])
plt.show()

#%% - Dự báo (forecast)
preds = fitted.get_forecast(len(test_data), alpha=0.05)
fc = preds.predicted_mean
fc.index = test_data.index
se, se.index = preds.se_mean, test_data.index
conf, conf.index = preds.conf_int(alpha=0.05), test_data.index
conf = conf.to_numpy()
lower_series = pd.Series(conf[:, 0], index=test_data.index)
upper_series = pd.Series(conf[:, 1], index=test_data.index)
fc_series=pd.Series(fc,index=test_data.index)
plt.plot(train_data,label="Training data")
plt.plot(test_data,color="green",label="Actual Traffic average month")
plt.plot(fc_series,color="red",label="Predicted Traffic average month")
plt.fill_between(lower_series.index, lower_series, upper_series, color="b", alpha=.10)
plt.title('Traffic average month prediction')
plt.xlabel("Date")
plt.ylabel("Traffic")
plt.legend()
plt.show()
