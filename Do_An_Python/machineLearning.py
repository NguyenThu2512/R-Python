#%%- import library
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#for preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling  import SMOTE

#for model score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score , recall_score , f1_score

#for models
import plotly.express as px
from sklearn import metrics
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import plot_roc_curve, roc_auc_score, roc_curve
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense , Dropout
from sklearn.svm import SVC
#%%nạp dữ liệu
df=pd.read_csv("./Data/Churn_Modelling.csv")
print(df.head())
print(df.shape)
print(df.info())
#%%- kiểm tra dữ liệu rỗng
df.isnull().sum()
#Kiểm tra kiểu dữ liêu
print(df.dtypes)
#Thống kê mô tả dữ liệu
print(df.describe())
#%%-Xóa các cột không liên quan
df.columns
df.drop(['RowNumber' ,'CustomerId' ,'Surname'] , axis =1 ,inplace = True)
df

#%%- Trực quan các biến phân loại
fig, ax = plt.subplots(3, 2)
sns.countplot('Geography', hue = 'Exited', data = df, ax = ax[0][0])
sns.countplot('Gender', hue = 'Exited', data = df, ax = ax[0][1])
sns.countplot('Tenure', hue = 'Exited', data = df, ax = ax[1][0])
sns.countplot('NumOfProducts', hue = 'Exited', data = df, ax = ax[1][1])
sns.countplot('HasCrCard', hue = 'Exited', data = df, ax = ax[2][0])
sns.countplot('IsActiveMember', hue = 'Exited', data = df, ax = ax[2][1])
plt.tight_layout()
plt.show()
#%% - Trực quan hóa các biến liên tục
fig, ax = plt.subplots(2, 2, figsize = (16, 10))
sns.boxplot(x = 'Exited', y = 'CreditScore', data = df, ax = ax[0][0])
sns.boxplot(x = 'Exited', y = 'Age', data = df, ax = ax[0][1])
sns.boxplot(x = 'Exited', y = 'Balance', data = df, ax = ax[1][0])
sns.boxplot(x = 'Exited', y = 'EstimatedSalary', data = df, ax = ax[1][1])
plt.tight_layout()
plt.show()

#%%- heatmap thể hiện mức tương quan giữa các biến
correlation = df.corr()
plt.figure(figsize = (12 ,12))
sns.heatmap(correlation, annot = True)
plt.show()
#%%- Lựa chọn biến mục tiêu
x=df.drop("Exited", axis='columns')
y=df['Exited']
df.info()
#%%- Xử lý giá trị phân loại
from sklearn.preprocessing import LabelEncoder
x['Geography_n']=LabelEncoder().fit_transform(x['Geography'])
x['Gender_n']=LabelEncoder().fit_transform(x['Gender'])
x_n=x.drop(["Geography", "Gender"],axis='columns')
x_n.info()
#%%- Xử lý mất cân bằng dữ liệu bằng smote
sns.countplot(df['Exited'])
plt.show()
smt = SMOTE()
x_res , y_res  = smt.fit_resample(x_n,y)
print(y_res.value_counts())

#%%- Tách dữ liệu train/test
x_train , x_test , y_train , y_test = train_test_split(x_res , y_res , test_size = 0.3 , random_state = 52)
print(x_train.shape , x_test.shape , y_train.shape , y_test.shape )
#%% - Scaling data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

#%%- create model using DecisionTree Classifier and fit training data
# support criteria are "gini" for the Gini index and "entropy" for the information gain
model=DecisionTreeClassifier(criterion='entropy',random_state=40).fit(x_train,y_train)
# create prediction
ytest_pred = model.predict(x_test)
print(ytest_pred[0:5])
#finding different scores

acc = accuracy_score(y_test ,ytest_pred)
print('DECISION TREE CLASSIFIER ACCURACY : ',acc)
pre = precision_score(y_test ,ytest_pred)
print('DECISION TREE CLASSIFIER PRECISION : ',pre)
rec = recall_score(y_test ,ytest_pred)
print('DECISION TREE CLASSIFIER RECALL : ',rec)
f1 = f1_score(y_test ,ytest_pred)
print('DECISION TREE CLASSIFIER F1_SCORE : ',f1)

#confution matrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, ytest_pred))
plt.title('Confution matrix' )
sns.heatmap(data = metrics.confusion_matrix(y_test , ytest_pred) , annot = True
            , fmt = '.0f', square = True)
plt.show()
# Mô hình dự đoán cho thấy có 1815 âm tính đúng, 519 dương tính giả, 469 âm tính giả và 1975 dương tính đúng
#%%- visualize result
features=["CreditScore", "Age", "Tenure","Balance", "NumOfProducts", "HasCrCard","IsActiveMember", "EstimatedSalary", "Geography", "Gender",]
text_representation=tree.export_text(model,feature_names=features)
print(text_representation)
#%%
plt.figure(figsize=(20,20), dpi=150)
t=tree.plot_tree(model, feature_names=features, class_names=['No', 'Yes'], filled=True)
plt.show()
#%% - Tính chính xác của dự báo trên tập Train và Test
score_train = model.score(x_train, y_train)
score_test = model.score(x_test, y_test)
print(score_train)
print(score_test)

#%% 10 Folds Cross Validation
clf_score = cross_val_score(model, x_train, y_train, cv=10)
print(clf_score)
print(clf_score.mean())


#%% Xác định mức quan trọng của các biến trong mô hình
f_imp = pd.Series(model.feature_importances_, index=x_n.columns.values).sort_values()
print(f_imp)
# visualize to see the feature importance
indices=np.argsort(f_imp)[::-1]
plt.figure(figsize=(20,10))
sns.barplot(x=f_imp, y = f_imp.index)
plt.show()

#%%- visualize result
features=["CreditScore", "Age", "Tenure","Balance", "NumOfProducts", "HasCrCard","IsActiveMember", "EstimatedSalary", "Geography", "Gender",]
text_representation=tree.export_text(model,feature_names=features)
print(text_representation)

#%%- create model using Random Forest Classifier and fit training data
rf = RandomForestClassifier().fit(x_train , y_train)

ytest_pred2 = rf.predict(x_test)
rf_Pred = pd.DataFrame.from_dict(ytest_pred2)
rf_Pred.head()
#%%finding different scores
acc2 = accuracy_score(y_test ,ytest_pred2)
pre2 = precision_score(y_test ,ytest_pred2)
rec2 = recall_score(y_test ,ytest_pred2)
f1_2 = f1_score(y_test ,ytest_pred2)
print('RANDOM FOREST  CLASSIFIER ACCURACY : ',acc2)
print('RANDOM FOREST  CLASSIFIER PRECISON : ',pre2)
print('RANDOM FOREST  CLASSIFIER RECALL : ',rec2)
print('RANDOM FOREST  CLASSIFIER F1_SCORE : ',f1_2)
#%% xác định tầm quan trọng của các biến trong mô hình ML
RF_feature_imp = pd.DataFrame(index=x_n.columns, data = rf.feature_importances_, columns = ['Importance']).sort_values("Importance", ascending = False)
RF_feature_imp
#trực quan hóa mức độ quan trọng
fig = px.bar(RF_feature_imp.sort_values('Importance', ascending = False), x = RF_feature_imp.sort_values('Importance',
             ascending = False).index, y = 'Importance', title = "Feature Importance",
             labels = dict(x = "Features", y ="Feature_Importance"))
fig.show()
#%% 10 Folds Cross Validation
clf_score2 = cross_val_score(rf, x_train, y_train, cv=10)
print(clf_score2)
print(clf_score2.mean())

#%% Trực quan ROC
# plot_roc_curve(rf, x_test, y_test)
rf_roc = plot_roc_curve(rf, x_test, y_test)
rf_roc.figure_.suptitle("ROC curve comparison")
plt.show()

#%% Xây dựng mô hình ANN
#Bây giờ là lúc xây dựng mô hình! Chúng ta sẽ sử dụng một mô hình tuần tự keras với ba lớp khác nhau gồm 2 lớp hidden và 1 lớp output
#Mô hình này đại diện cho một mạng nơ-ron chuyển tiếp (một mạng chuyển các giá trị từ trái sang phải). Chúng tôi sẽ chia nhỏ từng lớp và kiến ​​trúc của nó bên dưới.
# Model ANN sẽ có số notron lớp đầu nào (input layer) là 10, tướng ứng với số biến độc lập ,
# trong khi đó ở lớp đầu ra (output layer) số notron sẽ là 1
# Số notron và số lớp ẩn (hidden layer) thì chúng ta có thể tùy ý chọn
#
ann_model = Sequential()
ann_model.add(Dense(7, input_shape=(10,), activation='relu')) # Lớp ẩn thứ nhất
ann_model.add(Dense(5, activation='relu')) # Lớp ẩn thứ hai
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
ax.yaxis.set(ticks=(0,1), ticklabels=('Actual 0s', 'Predicted 1s'))
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
ax.yaxis.set(ticks=(0,1), ticklabels=('Actual 0s', 'Predicted 1s'))
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


