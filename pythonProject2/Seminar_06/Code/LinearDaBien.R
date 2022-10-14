install.packages("car")
install.packages("ggplot")
#Import library
library(plotly)
library(skimr)
library(zoo)
library(xts)
library(corrplot)
library(tidyr)
library(naniar)
library("Hmisc")
library("car") #package của function VIF
library(ggplot2)
#Load data
df <- read.csv('./data/Car_sales_cleaning_final.csv')
# %% Kiểm tra dữ liệu Null
miss_var_summary(df)
skimr::skim(df)
#%% Khai báo biến
price <- as.numeric(unlist(c(df["Price_in_thousands"])))
sale <-  as.numeric(unlist(c(df['Sales_in_thousands'])))
resale <- as.numeric(unlist(c(df['X__year_resale_value'])))
enginesize <- as.numeric(unlist(c(df['Engine_size'])))
horsepower <- as.numeric(unlist(c(df['Horsepower'])))
wheelbase <- as.numeric(unlist(c(df['Wheelbase'])))
width <- as.numeric(unlist(c(df['Width'])))
length <- as.numeric(unlist(c(df['Length'])))
curbweight <- as.numeric(unlist(c(df['Curb_weight'])))
fuelcapacity <- as.numeric(unlist(c(df['Fuel_capacity'])))
fuelefficiency <- as.numeric(unlist(c(df['Fuel_efficiency'])))
power <- as.numeric(unlist(c(df['Power_perf_factor'])))
#Chia tập dữ liệu Train và Split
sample <- sample(c(TRUE, FALSE), nrow(df), replace = T, prob = c(0.9,0.1))
train <- df[sample, ]
test <- df[!sample, ]
#Xây dựng model 1
model <- lm(price ~ sale + resale + enginesize + horsepower + wheelbase + width + length + curbweight + fuelcapacity + fuelefficiency + power , train)
summary(model)
#Xây dựng model 2
## Bỏ giá trị Resale
model <- lm(price ~ sale +  enginesize + horsepower + wheelbase + width + length + curbweight + fuelcapacity + fuelefficiency + power , train)
summary(model)
#Xây dựng model 3
## Bỏ giá trị Fuel capacity
model <- lm(price ~ sale +  enginesize + horsepower + wheelbase + width + length + curbweight +  power , train)
summary(model)
#Xây dựng model 4
## Bỏ giá trị Sale
model <- lm(price ~ enginesize + horsepower + wheelbase + width + length + curbweight +  power , train)
summary(model)
#Xây dựng model 5
## Bỏ giá trị Wheelbase
model <- lm(price ~ enginesize + horsepower +  width + length + curbweight +  power , train)
summary(model)
#Xây dựng model 6
## Bỏ giá trị Length
model <- lm(price ~ enginesize + horsepower +  width + curbweight +  power , train)
summary(model)
#Xây dựng model 7
## Bỏ giá trị curbweiht
model <- lm(price ~ enginesize + horsepower +  width +  power , train)
summary(model)
#Xây dựng model 8
## Bỏ giá trị width
model <- lm(price ~ enginesize + horsepower +   power , train)
summary(model)
## Nhìn vào mô hình có thể thấy 3 biến này đều có giá trị P value đều nhỏ hơn 0.05 (mức ý nghĩa 5%)
## Suy ra ba biến này có ý nghĩa về mặt thống kế đối với mô hình này
## Price = -0.75305 Enginesize - 0.90784 Horsepower + 2.57558 Power - 0.11283

# Kiểm định mô hình
# Giữa các biến độc lập không có mối quan hệ đa cộng tuyến hoàn hảo
vif(model)
# Theo Gujarati và Porter (2009) chỉ ra một số dấu hiệu của hiện tượng đa cộng tuyển trong mô hình khi:
# (1) VIF >= 10
# (2) Hệ số tương quan r của bất kì cắp biến nào trong mô hình lớn hơn 0.8
# Theo ta thấy thì giữa biến Horsepowwer và biến Power có sự đa cộng tuyến vô cùng lớn
# Xây dựng thêm hai mô hình giữa biến Enginesive với Horsepowwer và Enginesive với Power
#Enginesive với Horsepowwer
model <- lm(price ~ enginesize + horsepower , train)
summary(model)
vif(model)
# R square hiệu chỉnh là 71.85 % VIF đều bằng 3.3237215 đều nhỏ hơn 10
#Enginesive với Power
model <- lm(price ~ enginesize + power , train)
summary(model)
vif(model)
hist(price) # kiểm tra xem biến phụ thuộc của bạn có tuân theo phân phối chuẩn hay không .
# Sự phân bố của các quan sát gần như hình chuông, vì vậy chúng ta có thể tiến hành hồi quy tuyến tính.
# R square hiệu chỉnh là 83.68 % VIF đều bằng 3.01928 đều nhỏ hơn 10
# Kết luận: T có mô ình hồi quqy đa biến như sau:
# Price = -4.39093 Enginesize + 0.65903 Power - 10.03509
# Điều này có nghĩa là cứ tăng 1 đơn vị Enginesize, thì Giá xe (price) giảm 4.3903 đơn vị.
# Trong khi đó, cứ tăng 1 đơn vị Power thì Giá xe (Price) tăng 0.65903 đơn vị.
# plot(, pch=20, xlab = "Power", ylab = "Price", main = "Sactter of Power and Price")
# abline(model, col="red", lwd=3) # Vẽ đường tuyến tính trong mô hình
par(mfrow=c(2,2))
plot(model)
par(mfrow=c(1,1))
#Minh họa hồi quy cho mô hình
predicted_weight <- predict(model, data.frame(x = enginesize + power))
plot( enginesize + power ~ price ,data=train)
# abline(model , train), col="red", lwd=3) # Vẽ đường tuyến tính trong mô hình
#Dự báo
predict_values <- data.frame(power = c(200,150,300), enginesize = c(5,4.5,6))
pred_da <- predict(model, predict_values)

