#%%-import library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sms
from sklearn.linear_model import LinearRegression
#%%- Load data
df=pd.read_csv("data/cars.csv")

#%%-create model
x=df[['Weight','Volume']]
y=df[['CO2']]
model=LinearRegression().fit(x,y)


#%%- get results
R_sq=model.score(x,y)
intercept=model.intercept_
#intercept: la alpha
coefs=model.coef_
#R-sq: giai thich duoc 37% su bien dong ccua luong khi thao can cu vao trong luong va dung tich
#coefs: 0.0075: weight, 0.0078 volume

#%%-predict for present value
predicted_value=model.predict(x)
#the so truc tiep: alpha + ....
plt.plot(y,label='Actual')
plt.plot(predicted_value,label='predict', marker='x',color='r',linestyle='--')
plt.legend()
plt.show()

#%% - predict for future
x_new =[[1500,1300],[1400,1100]]
predicted_values=model.predict(x_new)

