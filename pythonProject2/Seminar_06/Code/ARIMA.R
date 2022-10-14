#%% - Import Library
library(urca)
library(ggplot2)
library(readr)
library(fpp2)
library(tidyverse)
library(forecast)
library(lubridate)
library(MLmetrics)
library(zoo)
library(plotly)
library(xts)
library(TSstudio)
library(tseries)

#%% - Read Data
df = read.csv('./data/portland-oregon-average-monthly-1.csv', header = TRUE)

#%% - Change format to date and rename
Rname <- df %>% select(c(Month,traffic))
colnames(Rname) <- c('Month','Average monthly ridership')
head(Rname)


#%% - Convert to times series data
Average_monthly_ridership <- ts(Rname$`Average monthly ridership`, start = c(1960,1), end = c(1969,6), frequency = 12)
#%% - Visualization Hist Plot
hist(Average_monthly_ridership, col="red", border =
"white",xlab="Average monthly ridership",ylab="Month", main = "Average monthly ridership of Portland")
#%% - visualization data
plot(Average_monthly_ridership, xlab="Time",ylab="Average monthly ridership", col = "red")


#%% - Decomposition of time series data
decomp <- decompose(Average_monthly_ridership)
plot(decomp, col= 'red')

#%% - Check stationary by using ADF test
adf.test(Average_monthly_ridership)
#p-value = 0.4864 > 0.05 => chuỗi chưa dừng

#%%
Average_monthly_ridership.DIFF <- diff(Average_monthly_ridership)
adf.test(Average_monthly_ridership.DIFF)
#p-value = 0.01 < 0.05 => chuỗi dừng

#%%
plot.ts(Average_monthly_ridership.DIFF, col="red")
#%% - ACF Plot with d =1
acf(as.numeric(Average_monthly_ridership.DIFF), main="Autocorrelation")
#%% - PACF Plot with d =1
pacf(as.numeric(Average_monthly_ridership.DIFF), main="Partial Autocorrelation")

#%% - Train data and Test data
train_data <- window(Average_monthly_ridership, start = c(1960,1), end = c(1968,6))
test_data <- window(Average_monthly_ridership, start = c(1968,7), end = c(1969,6))
plot(train_data, col = 'red')

#%% - Build ARIMA model
model <- auto.arima(train_data, approximation = FALSE, trace = TRUE)
summary(model)

#%% - Forecast
forecast_model <- forecast(model, h=12)

#%% - Prediction Plot
plot(forecast_model,col="red",ylab="Average monthly ridership",
main = "Forecast of Average monthly ridership")
forecast_model

#%% - Residuals Plot
hist(model$residuals, col="red", border = "white",
xlab="Residuals", main= "Histogram of Residuals")

#%% - Accuracy
accuracy(forecast_model,test_data)











