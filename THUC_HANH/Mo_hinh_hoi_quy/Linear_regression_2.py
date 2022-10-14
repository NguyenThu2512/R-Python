#%%-import library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sms
from sklearn.linear_model import LinearRegression
#%%- Load data
df=pd.read_csv("data/Sales.csv")

#%%-Create model
x=df[['Price','Advertising cost']]
y=df[['Sale']]
model=LinearRegression().fit(x,y)

#%% - Get results
R_sq=model.score(x,y)
intercept=model.intercept_
coefs=model.coef_

#%%- visualization
predict_value=model.predict(x)
plt.plot(y,label='Actual')
plt.plot(predict_value, label='Predict',color='r', marker='x', linestyle="--")
plt.legend()
plt.show()

#%%- predict for future value
x_new=[[4.2, 4.0],[4.8, 4.3]]
predict_values=model.predict(x_new)

#%%- split train/test dataset
from sklearn.model_selection import train_test_split
#phai theo thu tu dung
x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.1)


#%% - Create new model and fit it
new_model=LinearRegression().fit(x_train,y_train)
score=new_model.score(x_train,y_train)
print("R2; ", score)
print(y_test)
y_predd=new_model.predict(x_test)
print(y_predd)

