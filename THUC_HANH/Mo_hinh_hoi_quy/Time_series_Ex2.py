#%% - Nạp thư viện
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from pmdarima.arima import auto_arima
#%%- nạp dữ liệu
df=pd.read_csv("./data/Sales_Cars.csv", index_col="Month", parse_dates=True)
df.info()
#%%-ve bieu do the hien hoanh thu cua du lieu
plt.plot(df['Sales'])
plt.xlabel("Time")
plt.ylabel("Sales")
plt.show()
df_sales=(df['Sales'])
#%%-Phân rã chuỗi dữ liệu
#biểu đồ so sánh giá trị trung bình, độ lệch chuẩn và giá đóng cửa của 10 kì trước đó
rolmean=df_sales.rolling(3).mean()
rolstd=df_sales.rolling(3).std()
plt.plot(df_sales, label="original")
plt.plot(rolmean, label="Rolling mean")
plt.plot(rolstd, label="Rolling Std")
plt.legend()
plt.show()

#%%- Bieu do phan ra chuoi du liệu
decompose_result=seasonal_decompose(df_sales, period=10, model="multiplicative")
decompose_result.plot()
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
print(adf_test(df_sales))
print("----"*5)
print(kpss_test(df_sales))

#%%-kiem dinh tự tương quan (Auto correlation)
pd.plotting.lag_plot(df_sales)
plt.show()

#%%
plot_pacf(df_sales)
plt.show()
#%%
plot_acf(df_sales)
plt.show()
## Dữ liệu thể hiện không có sự tương quan giữa các kỳ và là chuỗi nhiễu trắng nên không thể thực hiện các bước tiếp theo