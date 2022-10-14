#%%- Nap thu vien
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.tsa.arima_model
import statsmodels.api as sm
from statsmodels.datasets import sunspots
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from pmdarima.arima import auto_arima

#%%-Nap du lieu phan tich
df=pd.read_csv("./data/ACG.csv",index_col="Date", parse_dates=True)
df.info()


#%% - Ve bieu do gia dong cua co phieu
plt.plot(df['Close'])
plt.xlabel("Date")
plt.ylabel("Close")
plt.show()


#%%- Chia tap du lieu huan luyen va kiem thu
df_close=np.log(df['Close'])
train_data, test_data=df_close[:int(len(df_close)*0.9)],df_close[int(len(df_close)*0.9):]
plt.plot(train_data,'blue', label='Train data')
plt.plot(test_data,'red',label="Test data")
plt.xlabel("Date")
plt.ylabel("Close price")
plt.legend()
plt.show()

#%%-Phan ra chuoi du lieu
#bieu do lich su so sanh gia dong cua vs gia tri trung binh va do lenhj chuan cuar 12 ki tu truoc do
rolmean=train_data.rolling(12).mean()
rolstd=train_data.rolling(12).std()
#lay ra 1 nguong nhat dinh de xem su tuong quan.
plt.plot(train_data,color="blue", label="original")
plt.plot(rolmean,color="red", label="rolling mean")
plt.plot(rolstd,color='green', label="rooling std")
plt.legend()
plt.show()

#%% Bieu do phan ra chuoi thoi gian

decompose_results = seasonal_decompose(train_data, model="multiplicative", period=30)
decompose_results.plot()
plt.show()

#%% Kiểm định tính dừng của dữ liệu (stationary)
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



#%%-kiem dinh tự tương quan (Auto correlation)
pd.plotting.lag_plot(train_data)
plt.show()
#%%
plot_pacf(train_data)
plt.show()

#%%
plot_acf(train_data)
plt.show()

#%% Chuyen doi du lieu ve chuoi dung
#Tinhs sai phan bac 1 du lieu train
# diff=train_data.diff(1)
diff=train_data.diff(1).dropna()
#Bieu do the hien du lieu ban dau va sau khi lay sai phan
fig, ax=plt.subplot(2,sharex="all")
train_data.plot(ax=ax[0],title= "Gia dong cua")
diff.plot(ax=ax[1], title="Sai phan bac nhat")
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
stepwise_fit=auto_arima(train_data, trace=True, suppress_warnings=True)
print(stepwise_fit.summary())
stepwise_fit.plot_diagnostics(figsize=(15,8))
plt.show()

#%%-create model
# model=ARIMA(train_data, order=(1,1,2))
# fitted=model.fit()
# print(fitted.summary())

model=sm.tsa.arima.ARIMA(train_data, order=(1,1,2))
fitted=model.fit()
print(fitted.summary())

#%% - Du bao(forecast)
preds=fitted.forecast(len(test_data), alpha=0.05)
preds.index = test_data.index
# fc, se, conf=fitted.forecast(len(test_data), alpha=0.05)
# fc_series=pd.Series(fc,index=test_data.index)
# lower_series=pd.Series(conf[:,0], index=test_data.index)
# upper_series=pd.Series(conf[:,1], index=test_data.index)
plt.figure(figsize=(16,10),dpi=150)
plt.plot(train_data,color="r",label="Training data")
plt.plot(test_data, color="orange", label="Actual stock price")
plt.plot(preds, color="b", label="Predicted stock price")
# plt.fill_between(lower_series.index, lower_series, upper_series, color="b", alpha=.10)
plt.title("Stock price prediction")
plt.xlabel("Time")
plt.ylabel("Stock price")
plt.legend()
plt.show()
