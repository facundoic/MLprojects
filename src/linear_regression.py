import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


path = '/home/facundoic/Desktop/GitHub/ML-repository/data/marathon-data/MarathonData.csv'

df_marathon = pd.read_csv(path)
df_marathon.drop(columns=['Name','id','Marathon','CATEGORY'],inplace=True)


df_marathon['CrossTraining'] = df_marathon['CrossTraining'].fillna(0)
df_marathon = df_marathon.dropna(how='any')

#print(df_marathon.isna().sum())
#! ver cuantos valores unicos tiene una columna
#print(df_marathon['CrossTraining'].unique())
#print(df_marathon['Category'].unique())

valores_cross = {"CrossTraining":{'ciclista 1h': 1,'ciclista 3h': 2,'ciclista 4h': 3,'ciclista 5h': 4,'ciclista 13h': 5}}
df_marathon.replace(valores_cross,inplace=True)


valores_category = {"Category":{'MAM':1,'M45':2,'M40':3,'M50':4,'M55':5,'WAM':6}}
df_marathon.replace(valores_category,inplace=True)

print(df_marathon.tail())

plt.scatter(x=df_marathon['km4week'],y=df_marathon['MarathonTime'])
plt.title('km4week vs Marathon Time')
plt.xlabel('km4week')
plt.ylabel('Marathon Time')
plt.show()

df_marathon = df_marathon.query('sp4week<1000')
plt.scatter(x=df_marathon['sp4week'],y=df_marathon['MarathonTime'])
plt.title('sp4week vs Marathon Time')
plt.xlabel('sp4week')
plt.ylabel('Marathon Time')
plt.show()

# ----------------------------------------------------------
df_train = df_marathon.sample(frac=0.8,random_state=0)
df_test = df_marathon.drop(df_train.index)

Y_train = df_train.pop('MarathonTime')
Y_test = df_test.pop('MarathonTime')

model = LinearRegression()
model.fit(df_train,Y_train)

predictions = model.predict(df_test)

error = np.sqrt(mean_squared_error(Y_test,predictions))
print("Error porcentual : %f" %(error*100))