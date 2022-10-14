#%% import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
#%% Load data
# Load data
df = pd.read_csv('./Data/Customertravel.csv')
#%% mã hóa dữ liệu
df['BookedHotel_int']= df.BookedHotelOrNot.map({'Yes':0, 'No': 1})
df['AccountSyncedToSocialMedia_int']= df.AccountSyncedToSocialMedia.map({'Yes':0, 'No': 1})
df['FrequentFlyer_int']= df.FrequentFlyer.map({'Yes':0, 'No': 1})
df['AnnualIncomeClass_int']= df.AnnualIncomeClass.map({'Low Income':1, 'Middle Income': 2, 'High Income':3})
#%% check null
df.isnull().sum()
#%% check duplicate
df.duplicated().sum()
#%%
df_remove = df.drop_duplicates(subset=["Age", "FrequentFlyer", "AnnualIncomeClass","ServicesOpted", "AccountSyncedToSocialMedia", "BookedHotelOrNot", "Target"], keep="first")
print(df_remove)
#%%
df_remove.info()

# #%% chia dữ liệu, XÉT BIẾN NHỊ PHÂN
# y = df_remove[['Target']]
# X = df_remove[['BookedHotel_int']]
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
# #%% model
# # Create model and fit it
# model = LogisticRegression(solver='liblinear', C=10.0, random_state=0).fit(X_train, y_train.values.ravel())
# #%%
# intercept = model.intercept_
# coefs = model.coef_
# score = model.score(X_train, y_train)
# print(score)
# print(coefs)

# #%% chia dữ liệu, XÉT BIẾN THỨ BẬC
# y = df_remove[['Target']]
# X = df_remove[['AnnualIncomeClass_int']]
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
# #%% Create model and fit it
# model = LogisticRegression(solver='liblinear', C=10.0, random_state=0).fit(X_train, y_train.values.ravel())
# #%%
# intercept = model.intercept_
# coefs = model.coef_
# score = model.score(X_train, y_train)
# print(score)
# print(coefs)

#%% chia dữ liệu, XÉT BIẾN NHỊ PHÂN
# y = df_remove[['Target']]
# X = df_remove[['AccountSyncedToSocialMedia_int']]
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
# #%% Create model and fit it
# model = LogisticRegression(solver='liblinear', C=10.0, random_state=0).fit(X_train, y_train.values.ravel())
# #%%
# intercept = model.intercept_
# coefs = model.coef_
# score = model.score(X_train, y_train)
# print(score)
# print(coefs)
#
#%% chia dữ liệu, XÉT BIẾN LIÊN TỤC
# y = df_remove[['Target']]
# X = df_remove[['ServicesOpted']]
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
# #%% Create model and fit it
# model = LogisticRegression(solver='liblinear', C=10.0, random_state=0).fit(X_train, y_train.values.ravel())
# #%%
# intercept = model.intercept_
# coefs = model.coef_
# score = model.score(X_train, y_train)
# print(score)
# print(coefs)

#%% chia dữ liệu, XÉT BIẾN LIÊN TỤC
# y = df_remove[['Target']].values.reshape(-1,1)
# X = df_remove[['Age']].values.reshape(-1,1)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
# #%% Create model and fit it
# model = LogisticRegression(solver='liblinear', C=10.0, random_state=0).fit(X_train, y_train.ravel())
# #%%
# intercept = model.intercept_
# coefs = model.coef_
# score = model.score(X_train, y_train)
# print(score)
# print(coefs)

#%% chia dữ liệu, XÉT BIẾN NHỊ PHÂN
y = df_remove[['Target']].values.reshape(-1,1)
X = df_remove[['FrequentFlyer_int']].values.reshape(-1,1)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
# X_train=np.arange(0,len(X_train),1)
#%% Create model and fit it
model = LogisticRegression(solver='liblinear', C=10.0, random_state=0).fit(x_train, y_train.ravel())
#%%
intercept = model.intercept_
coefs = model.coef_
score = model.score(x_train, y_train)
print(score)
print(coefs)
print(intercept)

# LẤY BIẾN NHỊ PHÂN FrequentFlyer vì có score lớn nhất (0.7447)
#%% figure
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['figure.dpi'] = 72
#%% headmap
sns.heatmap(np.round(df.corr(method ='spearman'), 2), annot=True,  cmap='Blues');
plt.show()


#%% ĐÁNH GIÁ MODEL
#kiểm tra hiệu suất của nó
prob_matrix = model.predict_proba(x_train)
y_train_pred = model.predict(x_train)
print(classification_report(y_train, y_train_pred, zero_division=0))
cm = confusion_matrix(y_train, y_train_pred)
#%%
#ma trận hỗn loạn
fig, ax =plt.subplots()
ax.imshow(cm)
ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[j, i], ha='center', va='center', color='r', fontsize=22)
plt.show()
#%%
plt.scatter(x_train, y_train, color='c', marker='o', label='Actual')
plt.scatter(x_train, y_train_pred, color='r', marker='+', label='Prediction')
plt.legend()
plt.show()

#%% making predictions
y_pred = model.predict(x_test)
pred_score = model.score(x_test, y_test)
pred_prob_matrix = model.predict_proba(x_test)
print(x_test)
pred_prob_matrix

#%% define prediction function via sigmoid
from my_funct import sigmoid
def prediction_function(age, inter, coef):
    z = inter + coef * age
    return sigmoid(z)

#%% Draw sigmoid plot
plt.scatter(x_train, y_train, color="r", marker='o')
X_test = np.linspace(-1, 4, 10)
sigs = []
for item in X_test:
    sigs.append(prediction_function(item, intercept[0], coefs[0][0]))
plt.plot(X_test, sigs, color='g', linestyle='--')
plt.scatter(x_test,  y_test, color='b', s=100, label='Actual')
plt.scatter(x_test, y_pred, color='y', marker='x', label='Predict')
plt.legend(loc='center right')
plt.show()

#%%
# pred_prob_matrix = prediction_function(56, intercept[0], coefs[0][0])
# print(pred_prob_matrix)
from sklearn.metrics import accuracy_score
print('The accuracy is :',accuracy_score(y_test, y_pred))

#%% 10 Folds Cross Validation
clf_score2 = cross_val_score(model, X_train, y_train, cv=10)
print(clf_score2)
print(clf_score2.mean())