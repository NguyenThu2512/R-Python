library(keras)
library(dplyr)
library(mlbench)
library(magrittr)
library(e1071)
library(ggplot2)
library('caret')
#Load data
df <- read.csv('./data/Churn_Modeling_final.csv')
str(df)
# df = subset(df, select = -c(df["Surname"]) )
df$Surname <- NULL
head(df)
#checking summary statistics
summary(df)
#checking missing values
sum(is.na(df))
# #%%
set.seed(1000)
split = sample.split(df$Exited, SplitRatio = 0.70)
# Split up the data using subset
training_data = subset(df, split==TRUE)
test_data = subset(df, split==FALSE)

#%% Chuyển đổi cột “Exited” của khung dữ liệu đào tạo thành một biến nhân tố.
training_data[["Exited"]] = factor(training_data[["Exited"]])
# Tạo model SVM
svm_model = svm(Exited ~ CreditScore + Geography_in + Gender_in + Age + Tenure + Balance + NumOfProducts + HasCrCard + IsActiveMember + EstimatedSalary,
                data = training_data, kernel = "linear", cost = 10, scale = FALSE)
svm_model
# print(svm_model)
# Dự đoán mô hình trên tập kiểm thử
y_pred = predict(svm_model, newdata = test_data)
y_pred
# Making the Confusion Matrix để dự đoán độ chính xác
confusionMatrix(table(y_pred, test_data$Exited))
