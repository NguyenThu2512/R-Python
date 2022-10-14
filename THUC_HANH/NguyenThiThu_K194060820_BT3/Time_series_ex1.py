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
#%%- nạp dữ liệu
df=pd.read_csv("./data/PNJ_2015_2020.csv", index_col="Date", parse_dates=True)
df.info()
#%%-ve bieu do gia dong cua cua dư lieu
plt.plot(df['Close'])
plt.xlabel("Time")
plt.ylabel("Close price")
plt.show()

#%%- Chia tập dữ liệu huấn luyện và kiểm thử
df_close=np.log(df['Close'])
df_train, df_test=df_close[:int(len(df_close)*0.9)], df_close[int(len(df_close)*0.9):]
plt.plot(df_train, label="Train data")
plt.plot(df_test, label="Test data")
plt.xlabel("Year")
plt.ylabel("Close price")
plt.legend()
plt.show()

#%%-Phân rã chuỗi dữ liệu
#biểu đồ so sánh giá trị trung bình, độ lệch chuẩn và giá đóng cửa của 10 kì trước đó
rolmean=df_train.rolling(10).mean()
rolstd=df_train.rolling(10).std()
plt.plot(df_train, label="original")
plt.plot(rolmean, label="Rolling mean")
plt.plot(rolstd, label="Rolling Std")
plt.legend()
plt.show()

#%%- Bieu do phan ra chuoi du liệu
decompose_results=seasonal_decompose(df_train, period=30, model="multiplicative")
decompose_results.plot()
plt.show()

#%%- Kiểm định tính dừng của dữ liệu
def adf_test(data):
    indices = ["ADF: Test statistic", "p value","H of lags", "H of observatión"]
    test = adfuller(data, autolag = "AIC")
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
print(adf_test(df_train))
print("----"*5)
print(kpss_test(df_train))

#%%-kiem dinh tự tương quan (Auto correlation)
pd.plotting.lag_plot(df_train)
plt.show()
#%%
plot_pacf(df_train)
plt.show()

#%%
plot_acf(df_train)
plt.show()

#%% Chuyen doi du lieu ve chuoi dung - Tinhs sai phan bac 1
diff=df_train.diff(1).dropna()
#Bieu do the hien du lieu ban dau va sau khi lay sai phan
fig, ax = plt.subplots(2,1, figsize=(6,4))
ax[0].plot(df_train)
ax[0].set_title("Gia dong cua")
ax[1].plot(diff)
ax[1].set_title("Sai phan bac nhat")
plt.show()

#%% - Kieem tra lai tinh dung cua du lieu sau khi lay sai phan
print(adf_test(diff))
print("------"*5)
print(kpss_test(diff))
plot_pacf(diff) #xac dinh tham so p cho mo hinh ARIMA
plt.show()

#%%
plot_acf(diff) #--> xac dinh tham so q cho mo hinh ARIMA
plt.show()


#%% - Xac dinh tham so p, d, q cho mo hinh ARIMA
stepwise_fit=auto_arima(df_train, trace=True, suppress_warnings=True)
print(stepwise_fit.summary())
stepwise_fit.plot_diagnostics(figsize=(15,8))
plt.show()

#%%-create model
model=ARIMA(df_train, order=(2,1,1))
fitted=model.fit()
print(fitted.summary())

#%% - Du bao(forecast)
preds=fitted.forecast(len(df_test),  alpha=0.05)
preds.index = df_test.index
plt.figure(figsize=(16,10),dpi=150)
plt.plot(df_train,color="r",label="Training data")
plt.plot(df_test, color="orange", label="Actual stock price")
plt.plot(preds, color="b", label="Predicted stock price")
# plt.fill_between(lower_series.index, lower_series, upper_series, color="b", alpha=.10)
plt.title("Stock price prediction")
plt.xlabel("Time")
plt.ylabel("Stock price")
plt.legend()
plt.show()


