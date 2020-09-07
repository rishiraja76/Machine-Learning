#Importing packages
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,confusion_matrix

#Reading the file
data=pd.read_csv("KNN_Project_Data")
print(data.head())

#Creating a pair plot
sns.pairplot(data=data,hue="TARGET CLASS")
plt.show()

#Scaling the data
scaler=StandardScaler()
scaler.fit(data.drop('TARGET CLASS',axis=1))
features = scaler.transform(data.drop('TARGET CLASS',axis=1))
df = pd.DataFrame(features,columns=data.columns[:-1])
print(df.head())

#Assigning the data to test and train the model with
X=features
y=data['TARGET CLASS']
X_train, X_test, y_train, y_test = train_test_split(X,y)

#Using a K Neighbors model on the data
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)

#Evaluating the performance
predictions = knn.predict(X_test)
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))

#Finding the best n_neighbors to use
error_rate = []
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))

#Creating a line plot
sns.lineplot(range(1,40),error_rate)
plt.show()

#Using a K Neighbors model on the data
knn = KNeighborsClassifier(n_neighbors=30)
knn.fit(X_train,y_train)

#Evaluating the performance
predictions = knn.predict(X_test)
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))