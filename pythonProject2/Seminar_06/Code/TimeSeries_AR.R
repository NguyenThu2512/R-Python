#%%- Import Library
# Library for data Processing
library(tidyverse)
# Library to forecast
library(forecast)
# Library for Processing date data type
library(lubridate)
# Library MAPE

library(MLmetrics)
# Library for Processing date data type
library(zoo)
# Data visualisation
library(plotly)
library(xts)
library(TSstudio)
library(tseries)
library(lubridate)
library(ggplot2)
library(urca)
#Read Data
df=read.csv('./Data/UNRATE.csv', header = TRUE)
# unemployment <- ts(df["U"], start = c(1948), frequency = 12)
head(df)
tail(df)
table(is.na(df))
summary(df)
table(is.na(df))
#rename
wine_venda <- df %>% select(c(M,U))
colnames(wine_venda) <- c('Month','Unemployment')
head(wine_venda)

#visualization data
unemployment <- ts(wine_venda$Unemployment, start=c(1948, 1), end=c(2020, 1), frequency=6)
plot(unemployment)

#Phân rã dữ liệu thời gian:
unemployment_deco <- decompose(unemployment, type = "multiplicative")
unemployment_deco$trend %>% autoplot()
unemployment_deco$seasonal %>% autoplot()

plot(decompose(unemployment))

#Kiểm tra chuỗi dừng bằng ADF test trong R và trực quan hoá
y_none=ur.df(unemployment,type = "none", selectlags = "AIC" )
summary(y_none)
y_drift=ur.df(unemployment,type="drift", selectlags = "AIC")
summary(y_drift)

y_trend=ur.df(unemployment,type="trend", selectlags = "AIC")
summary(y_trend)

adf.test(unemployment)
#Sử dụng giản đồ PACF và ACF

par(mfrow = c(1,2))
pacf(as.numeric(unemployment))
acf(as.numeric(unemployment))
#Sai phân bậc 1
em_diff <- diff(unemployment, differences = 1)
#Kiểm tra sau khi tính sai phân và chọn tham số d
pacf(as.numeric(em_diff))
acf(as.numeric(em_diff))
adf.test(em_diff)

#Chia tập dữ liệu train/test
#Train and test data
train_dat <- window(unemployment, start=c(1948,1), end=c(2020,1))
test_dat <- window(unemployment, start=c(1948,7), end=c(2020,1))

#Build model
model <- arima(train_dat , order = c(2, 0, 0))
summary(model)

#forcast
fcst <- forecast(model, h = 24)
fcst
#Call the point predictions
fcst$mean
#Plot the forecast
plot(fcst)