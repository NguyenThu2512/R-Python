#%% import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

#%% load data
from sklearn.preprocessing import StandardScaler
df = pd.read_csv('./Data/Customertravel.csv')
#%% mã hóa dữ liệu
df['BookedHotel_int']= df.BookedHotelOrNot.map({'Yes':0, 'No': 1})
df['AccountSyncedToSocialMedia_int']= df.AccountSyncedToSocialMedia.map({'Yes':0, 'No': 1})
df['FrequentFlyer_int']= df.FrequentFlyer.map({'Yes':0, 'No': 1})
df['AnnualIncomeClass_int']= df.AnnualIncomeClass.map({'Low Income':1, 'Middle Income': 2, 'High Income':3})
#%% clear data
df.info()
#%% check duplicate
df.duplicated().sum()
#%%
df_remove = df.drop_duplicates(subset=["Age", "FrequentFlyer", "AnnualIncomeClass","ServicesOpted", "AccountSyncedToSocialMedia", "BookedHotelOrNot", "Target"], keep="first")
print(df_remove)

#%%Kiểm tra sự phân phối của biến phụ thuộc
ax = df_remove.Target.value_counts().plot(kind='bar', color = ['r', 'lightblue'])
plt.show()

#%% heatmap
sns.heatmap(df_remove.corr(), annot=True, linewidths=.5, cmap="YlGnBu")
plt.title('Correlation between variables')
plt.show()

#%% chia tập dữ liệu
X = df_remove[['BookedHotel_int','AccountSyncedToSocialMedia_int', 'FrequentFlyer_int','AnnualIncomeClass_int']]
y = df_remove['Target']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)

#%%Chuẩn hóa các biến X dựa trên giá trị trung bình và độ lệch chuẩn.
SC=StandardScaler()
X_train=SC.fit_transform(X_train)
X_test=SC.fit_transform(X_test)
#%% tạo mô hình logistic
model_lr=LogisticRegression()
model_lr.fit(X_train,y_train)
y_pred=model_lr.predict(X_test)
print(model_lr.intercept_)
print(model_lr.coef_)

#%%Dùng các độ đo đánh giá mô hình
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))
#Độ chính xác của mô hình là 82.8% là trong 100 người dùng được dự đoán thì có khoảng 82 người được dự đoán đúng.

#%% configs
plt.rcParams['figure.figsize']=(10,8)
plt.rcParams['figure.dpi']=200
plt.rcParams['font.size']=13
#%% vẽ ma trận hỗn loạn
class_names=[0,1] # name of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(confusion_matrix(y_test,y_pred)), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()

#%% 10 Folds Cross Validation
clf_score2 = cross_val_score(model_lr, X_train, y_train, cv=10)
print(clf_score2)
print(clf_score2.mean())