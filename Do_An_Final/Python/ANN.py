#%% Import library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
# importing tensoflow libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense , Dropout
from tensorflow import keras
from imblearn.over_sampling  import SMOTE

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
#%% Xây dựng mô hình
#Bây giờ là lúc xây dựng mô hình! Chúng ta sẽ sử dụng một mô hình tuần tự keras với ba lớp khác nhau gồm 2 lớp hidden và 1 lớp output
#Mô hình này đại diện cho một mạng nơ-ron chuyển tiếp (một mạng chuyển các giá trị từ trái sang phải). Chúng tôi sẽ chia nhỏ từng lớp và kiến ​​trúc của nó bên dưới.
# Model ANN sẽ có số notron lớp đầu nào (input layer) là 10, tướng ứng với số biến độc lập ,
# trong khi đó ở lớp đầu ra (output layer) số notron sẽ là 1
# Số notron và số lớp ẩn (hidden layer) thì chúng ta có thể tùy ý chọn
#
ann_model = Sequential()
ann_model.add(Dense(7, input_shape=(10,), activation='relu')) # Lớp ẩn thứ nhất
ann_model.add(Dense(3, activation='relu')) # Lớp ẩn thứ hai
ann_model.add(Dense(1, activation='sigmoid')) # Lớp output
ann_model.summary()
#%% Compile the keras model Biên dịch với mô hình Keras
# Định nghĩa hàm mất mát (loss function), thuật toán tối ưu (optimizer), và phương thức so sánh hiệu năng của model
ann_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#%% Fit the keras model on the training dataset Huấn luyện ANN model
# my_model = ann_model.fit(x_train, y_train, epochs=100, validation_split = 0.33)
my_model = ann_model.fit(x_train, y_train, validation_split = 0.2, validation_data = (x_test, y_test), epochs = 100)
#%% Visualize
plt.figure(figsize = (12, 6))
train_loss = my_model.history['loss']
val_loss = my_model.history['val_loss']
epoch = range(1, 101)
sns.lineplot(epoch, train_loss, label = 'Training Loss')
sns.lineplot(epoch, val_loss, label = 'Validation Loss')
plt.title('Training and Validation Loss\n')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#Sai số (loss) ở tập test (Test set) và tập huấn luyện (Training set) không lớn lắm
#%%
plt.figure(figsize = (12, 6))
train_loss = my_model.history['accuracy']
val_loss = my_model.history['val_accuracy'] # của tập test
epoch = range(1, 101)
sns.lineplot(epoch, train_loss, label = 'Training accuracy')
sns.lineplot(epoch, val_loss, label = 'Validation accuracy') # Tập validation lấy từ Của tập test
plt.title('Training and Validation Accuracy\n')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
# Độ chính xác
#%% Lưu model
# ann_model.save('ANN Model')
#%% evaluate the keras model
acc = ann_model.evaluate(x_train, y_train)
print(f'Model in Train Data is {acc}')
acc = ann_model.evaluate(x_test, y_test)
print(f'Model in Test Data  is {acc}')
#%% Cross Validation
# ann_score = cross_val_score(ann_model, x_train, y_train, cv=10)
# print(ann_score)
# print(ann_score.mean())

#%% Dự đoán trên tập kiểm thử
y_pred = ann_model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)
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
print("Accuracy of ANN Model:",metrics.accuracy_score(y_test, y_pred))
# Model Precision: what percentage of positive tuples are labeled as such?
print("Precision of ANN Model:",metrics.precision_score(y_test, y_pred))
# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall of ANN Model:",metrics.recall_score(y_test, y_pred))

