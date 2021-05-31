import numpy as np # linear algebra
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression  # for Logistic Regression algorithm
from sklearn.model_selection import train_test_split #to split the dataset for training and testing
from sklearn.neighbors import KNeighborsClassifier  # for K nearest neighbours
from sklearn import svm  #for Support Vector Machine (SVM) Algorithm
from sklearn import metrics #for checking the model accuracy
from sklearn.tree import DecisionTreeClassifier #for using Decision Tree Algoithm
from sklearn.model_selection import cross_val_score,KFold,cross_validate


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

X = df[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
y = df.Species
X_train, X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=5)

models = []
models.append(('LR',LogisticRegression()))
models.append(('KNN',KNeighborsClassifier()))
models.append(('SVM',svm.SVC()))
models.append(('DECISION TREE',DecisionTreeClassifier()))

results = []
for name,model in models:
    kfold = KFold(n_splits=10)
    cv = cross_val_score(model,X_train,y_train,cv=kfold,scoring='accuracy')
    results.append((name,cv))
    print(name+' : ',cv.mean())

print(results)
