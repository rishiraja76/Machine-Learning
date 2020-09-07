#Importing packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import GridSearchCV

#Reading the file
iris = load_iris()

#Getting the data from the file and making it a dataframe
df_feat = pd.DataFrame(iris['data'],columns=iris['feature_names'])
print(df_feat.info())
df_target = pd.DataFrame(iris['target'],columns=['species'])
print(df_target.info())
df=pd.concat([df_feat,df_target],axis=1)
print(df.head())

#Creating a pair plot
sns.pairplot(df,hue="species")
plt.show()

#Creating a kde plot
sns.kdeplot(df[df["species"]==0]["sepal length (cm)"],df[df["species"]==0]["sepal width (cm)"])
plt.show()

#Assigning the data to test and train the model with
X=df.drop("species",axis=1)
y=df["species"]
X_train,X_test,y_train,y_test=train_test_split(X,y)

#Using a Support Vector model on the data
svc=SVC()
svc.fit(X_train,y_train)

#Evaluating the performance
predictions=svc.predict(X_test)
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))

#Using a Grid Search model on the data
param_grid={"C":[0.0001,0.001,0.01,0.1,1,10,100,1000],"gamma":[0.0001,0.001,0.01,0.1,1,10,100,1000]}
grid=GridSearchCV(SVC(),param_grid,verbose=5)
grid.fit(X_train,y_train)

#Evaluating the performance
pred=grid.predict(X_test)
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))
