library(keras)
library(dplyr)
library(mlbench)
library(neuralnet)
library(magrittr)
library(ggplot2)
library(ROCR)
library(caTools)
library(car)
library(caret)
library(CatEncoders)
library(e1071)

#Load data
df <- read.csv('./data/Churn_Modeling_final.csv')
str(df)
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


#%% Tạo model ANN
set.seed(333)
n <- neuralnet(Exited ~ CreditScore + Geography_in + Gender_in + Age + Tenure + Balance + NumOfProducts + HasCrCard + IsActiveMember + EstimatedSalary ,
               data = training_data,
               hidden = c(7,3),
               act.fct = "logistic",
                linear.output = FALSE)
#%% Vẽ biểu đồ mạng thần kinh
plot(n,col.hidden = 'darkgreen', col.hidden.synapse = 'darkgreen', show.weights = F,
     information = F,
     fill = 'lightblue')
#%% Dự đoán kết quả cho bộ thử nghiệm
y_pred = predict(n, newdata = test_data)
y_pred
## Chuyển đổi xác suất thành các lớp nhị phân.
pred <- ifelse(y_pred>0.5, 1, 0)
pred
# Making the Confusion Matrix để dự đoán độ chính xác
u <- union(pred, test_data$Exited)
t <- table(factor(pred, u), factor(test_data$Exited, u))
confusionMatrix(t)

