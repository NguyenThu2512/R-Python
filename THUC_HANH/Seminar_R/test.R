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

#Read Data
df = read.csv('./Data/portland-oregon-average-monthly-1.csv', header = TRUE)

#change format to date and rename
Rname <- df %>% select(c(Month,traffic))
colnames(Rname) <- c('Month','Average monthly ridership')
head(Rname)

# Convert to times series data and visualization data
Average_monthly_ridership <- ts(Rname$`Average monthly ridership`, start = c(1960,1),
                                end = c(1969,6), frequency = 12)
plot(Average_monthly_ridership)

#Decomposition of time series data
plot(decompose(Average_monthly_ridership))

