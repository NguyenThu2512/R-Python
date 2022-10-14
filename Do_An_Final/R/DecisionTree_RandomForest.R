#loading libraries
library(ggplot2)
library(ROCR)
library(dplyr)
library(caTools)
library(car)
library(caret)
library(CatEncoders)
library(e1071)
library(rpart)

library(repr)
library(randomForest)
library(keras)
library(dplyr)
library(mlbench)
library(neuralnet)
library(magrittr)
#loading Data
data<- read.csv("./Data/Churn_Modelling.csv")
str(data)
#checking summary statistics
summary(data)
#checking missing values
sum(is.na(data))
#plotting data to understand the context better
options(repr.plot.width=5, repr.plot.height=4)
ggplot(data,aes(x=Gender,y=Balance))+
  geom_boxplot(fill="#4271AE",alpha=0.7)+ ggtitle("Boxplot of Gender vs Balance")+theme_bw()


ggplot(data,aes(y=Age,x=Geography))+
  geom_boxplot(fill="#4271AE",alpha=0.7)+ ggtitle("Boxplot of Age vs Geography")+theme_bw()

#xóa cột không cần thiết và xử lý biến phân loại
data$RowNumber<-NULL
data$CustomerId<-NULL
data$Surname<- NULL
data$Gender<-as.factor(data$Gender)
data$Geography=as.factor(data$Geography)
str(data)
#splitting data into Train and Test sets
set.seed(1000)
split = sample.split(data$Exited, SplitRatio = 0.70)
# Split up the data using subset
train = subset(data, split==TRUE)
test = subset(data, split==FALSE)

##standardizing numerical variables
numattr<- subset(train,select = -c(Gender,Geography,Exited))
std_model <- preProcess(train[, colnames(numattr)], method = c("range"))
std_model
# The predict() function is used to standardize any other unseen data
train[,colnames(numattr)] <- predict(object = std_model, newdata = train[, colnames(numattr)])
test[, colnames(numattr)] <- predict(object = std_model, newdata = test[, colnames(numattr)])
str(train)


#create model
model=rpart(Exited~., data=train, method="class")
summary(model)
#prediction
predict=predict(model, newdata=test, type="class")
#confusion matrix
str(predict)
str(test$Exited)
x<-as.factor(predict)
y=as.factor(test$Exited)
confusionMatrix(x, y)


# Create object of importance of our variables


dt_importance <- varImp(model)
print(dt_importance)
# # Create plot of importance of variables
# ggplot(data = dt_importance, mapping = aes(x = dt_importance[,1])) +
#   geom_boxplot() +
#   labs(title = "Variable importance: Decision tree model") +
#   theme_light()
# fancyRpartPlot(model$finalModel, sub = '')
#visualize result
plot(model,main="Classification Tree for churn Class",
     margin=0.15,uniform=TRUE)
text(model,use.n=T)

#create random forest model
train$Exited=as.factor(train$Exited)
fit <- randomForest(Exited ~., data=train)
fit
print(fit)
#importance of each variable
importance(fit)
#predict
predict_value=predict(fit, newdata=test, type = "class")
#confusion matrix
x<-as.factor(predict_value)
test$Exited<-as.factor(test$Exited)
str(predict_value)
str(test$Exited)
confusionMatrix(x, test$Exited)

