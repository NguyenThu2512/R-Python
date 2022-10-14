#%%-import library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sms

#%%- Load data
df=pd.read_csv("data/Income.csv")
income=df['Income']
expenditure=df['Expenditure']
income,expenditure=np.array(income),np.array(expenditure)
income=sms.add_constant(income)
#%%- Create model
model=sms.OLS(expenditure,income)
results=model.fit()

#%%- Get results
R_square=results.rsquared
params=results.params


#%%-Predict
pre_value=params[0]+params[1] * df['Income']

#%%-Visual
plt.plot(df['Income'],df['Expenditure'],lable='Actual')
plt.plot(df['Income'],pre_value,color='r', linestyle='--', marker='x', label='predict')
plt.xlabel("Income")
plt.ylabel("Expenditure")
plt.legend()
plt.show()

#%%-predict future value
other_income=np.array([26,28,31])
other_income=sms.add_constant(other_income)
pre_future_values=results.predict(other_income)