# đa biến
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
#load data
getwd()
df <- read.csv("./Data/Customertravel_Clean.csv")
str(df)
summary(df)
#kiểm tra thông tin dữ liệu
miss_var_summary(df)
skimr::skim(df)
# chia data
set.seed(150) #giúp ngẫu nhiên có hệ thống,
# khi có random seed thì nó sẽ giữ kết quả random của mình, coi như dữ liệu training của mình là cố định, để có thể đánh giá các kết quả phía dưới
split <- sample.split(df$Target, SplitRatio <- 0.75) #sample là hàm random
training <- subset(df, split == TRUE)
test <- subset(df, split == FALSE)
#Create model, để biến
model <- glm( Target ~ FrequentFlyer + Age + AnnualIncomeClass +ServicesOpted + AccountSyncedToSocialMedia + BookedHotelOr0t,
                data = training, family = binomial)
summary(model)$coef
#hệ số chặn là intercept, vd frequenlyt hệ số là 2.52...
# xác định biến nào tương quan + hay -, p value >0.05 thì loại, <0.05 thì lấy
#tính thêm hàm AIC để xác định
#Dự đoán xác suất
probabilities <- model %>% predict(test, type = "response")
head(probabilities)
# biểu đồ ROC...

# dự đoán các lớp
predicted.classes <- ifelse(probabilities > 0.5, "1", "0")
head(predicted.classes)
#Đánh giá độ chính xác của mô hình
mean(predicted.classes == test$Target)
