#Importing packages
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

#Reading the file
ad_data=pd.read_csv("advertising.csv")

#Printing information about the file
print(ad_data.head())
print(ad_data.info())
print(ad_data.describe())

#Creating a histogram
sns.distplot(ad_data["Age"],kde=False)
plt.show()

#Creating a joint plot
sns.jointplot(x="Area Income",y="Age",data=ad_data)
plt.show()

#Creating a joint plot
sns.jointplot(x="Daily Time Spent on Site",y="Age",data=ad_data,kind="kde")
plt.show()

#Creating a joint plot
sns.jointplot(x="Daily Time Spent on Site",y="Daily Internet Usage",data=ad_data)
plt.show()

#Creating a pair plot
sns.pairplot(data=ad_data,hue="Clicked on Ad")
plt.show()

#Assigning the data to test and train the model with
X=ad_data[["Daily Time Spent on Site","Age","Area Income","Daily Internet Usage","Male"]]
y=ad_data["Clicked on Ad"]
X_train, X_test, y_train, y_test=train_test_split(X,y)

#Using a Logistic Regression model on the data
logmodel= LogisticRegression()
logmodel.fit(X_train,y_train)

#Evaluating the performance
predictions= logmodel.predict(X_test)
print(classification_report(y_test,predictions))