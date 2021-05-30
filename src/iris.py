import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression  # for Logistic Regression algorithm
from sklearn.model_selection import train_test_split #to split the dataset for training and testing
from sklearn.neighbors import KNeighborsClassifier  # for K nearest neighbours
from sklearn import svm  #for Support Vector Machine (SVM) Algorithm
from sklearn import metrics #for checking the model accuracy
from sklearn.tree import DecisionTreeClassifier #for using Decision Tree Algoithm

path = '/home/facundoic/Desktop/GitHub/ML-repository/MLprojects/data/iris-data/Iris.csv'
df = pd.read_csv(path)
df.set_index("Id",inplace=True)

print(df.head())
print(df.info())
print(df['Species'].unique())
#This graph show the relationship between the sepal length and width
figure1 = df[df.Species == 'Iris-setosa'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='orange',label='Setosa')
df[df.Species == 'Iris-versicolor'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='pink',label='Versicolor',ax=figure1)
df[df.Species == 'Iris-virginica'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='green',label='Virginica',ax=figure1)
figure1.set_title('Sepal Lenght vs Width')
plt.show()
#This graph show the elationship between the petal length and width
figure2 = df[df.Species == 'Iris-setosa'].plot(kind='scatter',x='PetalLengthCm',y='PetalWidthCm',color='orange',label='Setosa')
df[df.Species == 'Iris-versicolor'].plot(kind='scatter',x='PetalLengthCm',y='PetalWidthCm',color='pink',label='Versicolor',ax=figure2)
df[df.Species == 'Iris-virginica'].plot(kind='scatter',x='PetalLengthCm',y='PetalWidthCm',color='green',label='Virginica',ax=figure2)
figure2.set_title('Petal Lenght vs Width')
plt.show()

train,test = train_test_split(df,test_size=0.25)
print(train.shape)
print(test.shape)
#!-------------------------------------------------------------------------------------
X_train = train[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
y_train = train.Species

X_test = test[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
y_test = test.Species
#!------------------------------SVM -------------------------------------------------------
model_svm  = svm.SVC()
model_svm.fit(X_train,y_train)
prediction_svm = model_svm.predict(X_test)
print('The accuracy of the SVM is : ',metrics.accuracy_score(prediction_svm,y_test))
#!-------------------------------LOGISTIC REGRESSION-----------------------------------------------------
model_lr = LogisticRegression()
model_lr.fit(X_train,y_train)
prediction_lr = model_lr.predict(X_test)
print('The accuracy of the Logicistic Regression is: ',metrics.accuracy_score(prediction_lr,y_test))
#!--------------------------------DECISION TREE-----------------------------------------------------------
model_dt = DecisionTreeClassifier()
model_dt.fit(X_train,y_train)
prediction_dt = model_dt.predict(X_test)
print('The accuracy of Decision Tree is: ',metrics.accuracy_score(prediction_dt,y_test))
#!------------------------------------K-NEAREST NEIGHBOURS-------------------------------------------------
model_knn = KNeighborsClassifier(n_neighbors=3)
model_knn.fit(X_train,y_train)
prediction_knn = model_knn.predict(X_test)
print('The accuracy of the KNN is: ',metrics.accuracy_score(prediction_knn,y_test))
#!-------------------------------------------KNN various values---------------------------------------------
"""
index = list(range(1,11))
result = pd.Series()
x =[i for i in range(1,11)]

for i in index:
    model_knn2 = KNeighborsClassifier(n_neighbors=i)
    model_knn2.fit(X_train,y_train)
    prediction_knn2 = model_knn2.predict(X_test)
    result.append(pd.Series(metrics.accuracy_score(prediction_knn2,y_test)))
plt.plot(index,result)
plt.xticks(x)
plt.show()
"""