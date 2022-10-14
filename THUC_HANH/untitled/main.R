#Import Library
library(readr)
library(ggplot2)
library(fpp2)
library(tseries)
library(forecast)
#Read Data
df=read.csv('./data/Electric_Production.csv', header =
TRUE)
