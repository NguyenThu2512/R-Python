# đơn biến
# library
library(tidyverse)
library(caret)
library(DataExplorer)
library(caTools)
library(GGally)
library(ggplot2)
library(skimr)
library(xts)
library(corrplot)
library(tidyr)
library(naniar)
library("Hmisc")

#load data
getwd()
df <- read.csv("./Data/Customertravel_Clean.csv")
str(df)
summary(df)
#kiểm tra thông tin dữ liệu
miss_var_summary(df)
skimr::skim(df)
#tương quan dữ liệu
plot_correlation(na.omit(df), maxcat = 5L)
# nhìn vào biểu đồ ta có thể thấy đc sự tương quan giữa biến Target và FrequentFlyer
# Chia tập train, test với tỉ lệ 90-10
sample <- sample(c(TRUE, FALSE), nrow(df), replace = T, prob = c(0.9,0.1))
train <- df[sample, ]
test <- df[!sample, ]
#Xây dựng model hồi quy logistic thể hiện mối quan hệ giữa Target và FrequentFlyer
model <- glm(Target ~FrequentFlyer, data=train, family = binomial)
summary(model)
# predict
probabilities=predict(model,type="response")
# Look at the first five
probabilities[1:5]
# kiểm tra độ chính xác của model
fitted.results <- predict(model,newdata=subset(test,select=c(2,3)),type='response')
fitted.results <- ifelse(fitted.results > 0.5,1,0)
misClasificError <- mean(fitted.results != test$Target)
print(paste('Accuracy',1-misClasificError))
# ma trận hỗn loạn
install.packages("e1071")
confusionMatrix(data=as.factor(fitted.results), reference=as.factor(test$Target))
