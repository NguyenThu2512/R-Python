#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC
warnings.filterwarnings("ignore")
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
#%% Load data
df = pd.read_csv('./data/Churn_Modelling.csv')
#%%
df.info()
#%% #kiểm tra dữ liệu rỗng
df.isnull().sum()
#%%#Thống kê mô tả dữ liệu
print(df.describe())
#%%-Xóa các cột không liên quan
df.columns
df.drop(['RowNumber' ,'CustomerId' ,'Surname'] , axis =1 ,inplace = True)
#%%- heatmap thể hiện mức tương quan giữa các biến
correlation = df.corr()
plt.figure(figsize = (12 ,12))
sns.heatmap(correlation, annot = True)
plt.show()
#%% Chuyển đổi biến phân loại
df['Geography_in']=df.Geography.map({'France':0,'Spain':1,'Germany':2})
df['Gender_in']=df.Gender.map({'Female':0,'Male':1})
#%%
df = df.drop(["Geography", "Gender"], axis = 1)
#%%
x = df[['CreditScore', 'Geography_in','Gender_in', 'Age', 'Tenure','Balance','NumOfProducts','HasCrCard','IsActiveMember','EstimatedSalary' ]]
y = df[['Exited']]
#%%- Xử lý mất cân bằng dữ liệu bằng smote
sns.countplot(df['Exited'])
plt.show()

#%%- Xử lý mất cân bằng dữ liệu bằng smote
sns.countplot(df['Exited'])
plt.show()
smt = SMOTE()
x_res , y_res  = smt.fit_resample(x,y)
print(y_res.value_counts())

#%%- Tách dữ liệu train/test
x_train , x_test , y_train , y_test = train_test_split(x_res , y_res , test_size = 0.3 , random_state = 52)
print(x_train.shape , x_test.shape , y_train.shape , y_test.shape )
#%% Co giãn dữ liệu
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
#%% Tạo model SVM
svm_model = SVC() # Linear Kernel
#Train the model using the training sets
svm_model.fit(x_train, y_train)
#%%# Độ chính xác (accuracy) trên tập huấn luyện (training set)
svm_model_score = svm_model.score(x_train, y_train)
print("Độ chính xác (accuracy) trên tập huấn luyện", svm_model_score)
# Độ chính xác (accuracy) trên tập kiểm thử (test set)
svm_model_score_val = svm_model.score(x_test, y_test)
svm_model_score_val
print("Độ chính xác (accuracy) trên tập kiểm thử", svm_model_score_val)
#Độ chính xác (accuracy) của model trên cả tập huấn luyện (training set) và tập test (test set) là gần như tương đồng nhau, cho thấy mức độ tổng quát (generalization) tốt của SVM model đã được huấn luyện.
#%% Dự đoán trên tập kiểm thử
y_pred = svm_model.predict(x_test)
print(y_pred)
#%%
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred) #ma trận hỗn loạn
print(cm)
#%% Ma trận hỗn loạn
fig, ax = plt.subplots()
ax.imshow(cm)
ax.xaxis.set(ticks=(0,1), ticklabels=('Predicted 0s', 'Predicted 1s'))
ax.yaxis.set(ticks=(0,1), ticklabels=('Actual 0s', 'Actual 1s'))
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i,j], ha='center', va='center',color='r', fontsize = 22)
plt.show()
#%% In ra các chỉ số Precision, Recall
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
print(classification_report(y_test, y_pred))
#%%
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy: how often is the classifier correct?
print("Accuracy of SVM Model:",metrics.accuracy_score(y_test, y_pred))
# Model Precision: what percentage of positive tuples are labeled as such?
print("Precision of SVM Model:",metrics.precision_score(y_test, y_pred))
# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall of SVM Model:",metrics.recall_score(y_test, y_pred))

