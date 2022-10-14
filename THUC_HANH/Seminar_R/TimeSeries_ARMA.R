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
df=read.csv('./Data/airline-passenger-traffic(1).csv', header = TRUE)
head(df)
tail(df)
table(is.na(df))
summary(df)

# rename
traffic <- df %>% select(c(M,P))
colnames(traffic) <- c('Month','Passenger')
head(traffic)

#Xử lý missing data bằng việc loại bỏ
traffic2 <- traffic %>%
  drop_na(Month,Passenger)
summary(traffic2)

#visualization data
passenger <- ts(traffic2$Passenger, start=c(1948, 1), end=c(1960, 1), frequency=6)
plot(passenger)

#Phân rã dữ liệu thời gian:
passenger_deco <- decompose(myts, type = "multiplicative")
passenger_deco$trend %>% autoplot()
passenger_deco$seasonal %>% autoplot()

plot(decompose(passenger))

#Kiểm tra chuỗi dừng bằng ADF test trong R và trực quan hoá
# y_none=ur.df(passenger,type = "none", selectlags = "AIC" )
# summary(y_none)
# y_drift=ur.df(passenger,type="drift", selectlags = "AIC")
# summary(y_drift)
# y_trend=ur.df(passenger,type="trend", selectlags = "AIC")
# summary(y_trend)
#
# adf.test(passenger)
#Sử dụng giản đồ PACF và ACF tìm (p,q)
par(mfrow = c(1,2))
pacf(as.numeric(passenger))
acf(as.numeric(passenger))



# #Build model
# model <- arima(passenger , order = c(1, 0, 1))
# summary(model)
# #forcast
# fcst <- forecast(model, h = 10)
# fcst
# #Call the point predictions
# fcst$mean
# #Plot the forecast
# plot(fcst)
#Xây dựng model AR
AR <- arima(passenger, order = c(1,0,0))
print(AR)
ts.plot(passenger)
AR_fit <- passenger - residuals(AR)
points(AR_fit, type = "l", col = 2, lty = 2)
# Sử dụng predict để dự đoán
predict_AR <- predict(AR)
predict_AR$pred[1]
predict(AR, n.ahead = 10)
#Vẽ biểu đồ chuỗi AirPassenger cộng với khoảng thời gian dự đoán và dự đoán 95%.
ts.plot(passenger, xlim = c(1949, 1961))
AR_forecast <- predict(AR, n.ahead = 10)$pred
AR_forecast_se <- predict(AR, n.ahead = 10)$se
points(AR_forecast, type = "l", col = 2)
points(AR_forecast - 2*AR_forecast_se, type = "l", col = 2, lty = 2)
points(AR_forecast + 2*AR_forecast_se, type = "l", col = 2, lty = 2)
#Xây dựng model MA để dự đoán
MA <- arima(passenger, order = c(0,0,1))
print(MA)
ts.plot(passenger)
MA_fit <- passenger - resid(MA)
points(MA_fit, type = "l", col = 2, lty = 2)
#Predict with MA model
predict_MA <- predict(MA)
predict_MA$pred[1]
predict(MA,n.ahead=10)
ts.plot(passenger, xlim = c(1949, 1961))
MA_forecasts <- predict(MA, n.ahead = 10)$pred
MA_forecast_se <- predict(MA, n.ahead = 10)$se
points(MA_forecasts, type = "l", col = 2)
points(MA_forecasts - 2*MA_forecast_se, type = "l", col = 2, lty = 2)
points(MA_forecasts + 2*MA_forecast_se, type = "l", col = 2, lty = 2)
# Find correlation between AR_fit and MA_fit
cor(AR_fit, MA_fit)
# Find AIC of MA
AIC(AR)
# Find AIC of MA
AIC(MA)
# Find BIC of AR
BIC(AR)
# Find BIC of MA
BIC(MA)
#Với mô hình AR cho giá trị AIC và BIC nên ta chọn mô hình AR