#%% -Import library
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as pd
warnings.filterwarnings('ignore')

#%%-some config
plt.rcParams['figure.figsize']=(10,8)
plt.rcParams['figure.dpi']=100
plt.rcParams['font.size']=14


#%% - Load data
df=pd.read_csv('data/Income.csv')
income=df[["Income"]]
expenditure=df[["Expenditure"]]

#%% - Visualization
plt.scatter(income,expenditure)
plt.show()
plt.xlabel("Income")
plt.ylabel("Expenditure")


#%%- Create model
model=LinearRegression().fit(income,expenditure)


#%%-Get result
intercept=model.intercept_
slope=model.coef_
R_square=model.score(income,expenditure)

#%%-

#%%-Prediction
predicted_value=model.predict(income)

#%%- Future Prediction
# future_value=np.array([26,28,31]).reshape(-1,1)
# #Cach1: predicted_value=intercept+slope*future_value
# prdicted_value=model.predict(future_value)


#%%- visualization
plt.plot(income,expenditure,color='r', label='Actual')
plt.plot(income,predicted_value,color='g',linestyle='--', marker='x', label='Predict')
plt.xlabel("Income")
plt.ylabel("Expenditure")
plt.legend()
plt.show()
