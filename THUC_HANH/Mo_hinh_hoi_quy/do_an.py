#%%-import library
import numpy as pd
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#%% - import data
df=pd.read_csv("data/CarPrice_Assignment.csv")
price=df[["price"]]
wheelbase=df[["wheelbase"]]
carlength=df[["carlength"]]
carwidth=df[["carwidth"]]
carheight=df[["carheight"]]
curbweight=df[["curbweight"]]
enginesize=df[["enginesize"]]
horsepower=df[["horsepower"]]


#%%-Visualization
def scatter(x,fig):
    plt.subplot(4,3,fig)
    plt.scatter(df[[x]],price)
    plt.title(x + ' and price')
    plt.xlabel('price')
    plt.ylabel(x)
plt.figure(figsize=(20,20))
scatter("wheelbase",1)
scatter("carlength",2)
scatter("carwidth",3)
scatter("curbweight",4)
scatter("enginesize",5)
scatter("horsepower",6)
scatter("carheight",7)
scatter("boreratio",8)
scatter("stroke",9)
scatter("compressionratio",10)
scatter("citympg",11)
scatter("highwaympg",12)
plt.show()
df_car = df[['price', 'fueltype', 'aspiration','carbody', 'drivewheel','wheelbase',
                  'curbweight', 'enginetype', 'cylindernumber', 'enginesize', 'boreratio','horsepower',
                    'fueleconomy', 'carlength','carwidth', 'carsrange']]



#%%-create model
model1=LinearRegression().fit(curbweight,price)

#%%-get result
intercept=model1.intercept_
slope=model1.coef_
R_square=model1.score(curbweight,price)

#%%-prediction
predicted_value=model1.predict(curbweight)

#%%-visualization
plt.plot(df['price'],linestyle='--', color='g', label='Actual')
plt.plot(predicted_value,linestyle='--', marker='x',color='b',label='predict')
plt.xlabel("price")
plt.ylabel("curbweight")
plt.legend()
plt.show()
