# đa biến
# library
library(tidyverse)
library(caret)
library(DataExplorer)
library(caTools)
library(GGally)
library(ggplot2)
#load data
getwd()
df <- read.csv("./Data/Customertravel_Clean.csv")
str(df)
summary(df)
# chia data
set.seed(150)
split <- sample.split(df$Target, SplitRatio <- 0.75)
training <- subset(df, split == TRUE)
test <- subset(df, split == FALSE)
#Create model
model <- glm( Target ~ FrequentFlyer + Age + AnnualIncomeClass +ServicesOpted + AccountSyncedToSocialMedia + BookedHotelOr0t,
                data = training, family = binomial)
summary(model)$coef

#Dự đoán xác suất
probabilities <- model %>% predict(test, type = "response")
head(probabilities)
# dự đoán các lớp
predicted.classes <- ifelse(probabilities > 0.5, "1", "0")
head(predicted.classes)
#Đánh giá độ chính xác của mô hình
mean(predicted.classes == test$Target)
