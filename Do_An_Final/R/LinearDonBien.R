#Import library
library(plotly)
library(skimr)
library(zoo)
library(xts)
library(corrplot)
library(tidyr)
library(naniar)
library("Hmisc")
#Load data
df <- read.csv('./data/Car_sales_cleaning_final.csv')
# %% Kiểm tra dữ liệu Null
miss_var_summary(df)
skimr::skim(df)
# %% Xem chi tiết tập dữ liệu
# print(df.describe())
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
# typeCar = df['Vehicle_type_Car'].values.reshape(-1,1)
# typePassenger = df['Vehicle_type_Passenger'].values.reshape(-1,1)

# Trực quan bằng biểu đò
par(mar=c(1,1,1,1))
par(mfrow=c(2,6))
plot(sale, price, pch=16, xlab = "Sales", ylab = "Price", main = "Sactter of Sale and Price" )
plot(resale, price,pch=16, xlab = "Resale value", ylab = "Price", main = "Sactter of Resale value and Price" )
plot(enginesize, price,pch=16, xlab = "Enginesize", ylab = "Price", main = "Sactter of Enginesize and Price" )
plot(horsepower, price,pch=16, xlab = "Horsepower", ylab = "Price", main = "Sactter of Horsepower and Price" )
plot(wheelbase, price,pch=16, xlab = "Wheelbase", ylab = "Price", main = "Sactter of Wheelbase and Price" )
plot(width, price,pch=16, xlab = "Width", ylab = "Price", main = "Sactter of Width and Price" )
plot(length, price,pch=16, xlab = "Length", ylab = "Price", main = "Sactter of Length and Price" )
plot(curbweight,pch=16, price, xlab = "Curbweight", ylab = "Price", main = "Sactter of Curbweight and Price" )
plot(fuelcapacity,pch=16, price,xlab = "Fuelcapacity", ylab = "Price", main = "Sactter of Fuelcapacity and Price" )
plot(fuelefficiency,pch=16, price,xlab = "Fuelefficiency", ylab = "Price", main = "Sactter of Fuelefficiency and Price" )
plot(power, price,pch=16,xlab = "Power", ylab = "Price", main = "Sactter of Power and Price" )
par(mfrow=c(1,1))
# xác định sự tương quan giữa các biến thông qua sơ đồ heatmap
x <- data.frame(price, sale, enginesize, horsepower, wheelbase, width, length, curbweight, fuelcapacity, fuelefficiency, power)
print(x)
correlation <- cor(x, y = NULL, use = "everything", method = c("pearson"))
corrplot(correlation, method = "number")
# Xác định các hệ số R_squảre
# Nhìn vào  biểu đồ ta có sự tương quan giữa các biến với giá như sau:
# Sale: -0.31 tương quan âm và tương quan yếu
#Enginesize: 0.63 tương quan dương và tương quan trung bình
#Horsepowwer 0.84 tương quan dương và tương quan mạnh
# Wheelbase: 0.11 tương quan dương và tương quan yếu
#Width 0.33 tương quan dương và tương quan yếu
#Length 0.16 tương quan dương, tương quan yếu
#Curbweight: 0.53 tương quan dương, tương quan trung bình
#Fuelcapacity; 0.42 tương quan dương, tương quan yếu
#Fuelefficency: -0,49 tương quan âm, tương quan trung bình
#Power: 0.9 tương quan dương, tương quan mạnh

#Nhìn vào các hệ số trên chúng ta có thể lựa chọn được biến Power có mối tương quan với Prices
#Xây dựng mô hình:
model <- lm(price ~ power, df)
#Sử dụng phương pháp bình phương nhỏ nhất OLS (Ordinary least squares)
summary(model)
#Mô hình giải thích được 80%
# prices = -11.9589 + 0.5099 Power

#Vẽ biểu đồ minh họa mô hình hồi quy đơn biến
plot(power, price, pch=20, xlab = "Power", ylab = "Price", main = "Sactter of Power and Price")
abline(model, col="red", lwd=3) # Vẽ đường tuyến tính trong mô hình
# Dự báo
predict_values <- data.frame(power = c(200,150,300))
pred <- predict.lm(model, predict_values)







