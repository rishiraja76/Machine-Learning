#Importing packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.ensemble import RandomForestClassifier

#Reading the file
loans=pd.read_csv("loan_data.csv")

#Printing information about the file
print(loans.info())
print(loans.head())
print(loans.describe())

#Creating a histogram
loans[loans['credit.policy']==1]['fico'].hist(label='Credit.Policy=1')
loans[loans['credit.policy']==0]['fico'].hist(label='Credit.Policy=0')
plt.legend()
plt.xlabel('FICO')
plt.show()

#Creating a histogram
loans[loans['not.fully.paid']==1]['fico'].hist(label='not.fully.paid=1')
loans[loans['not.fully.paid']==0]['fico'].hist(label='not.fully.paid=0')
plt.legend()
plt.xlabel('FICO')
plt.show()

#Creating a count plot
sns.countplot(x="purpose",data=loans,hue="not.fully.paid")
plt.show()

#Creating a joint plot
sns.jointplot(x="fico",y="int.rate",data=loans)
plt.show()

#Creating a linear relationship plot
sns.lmplot(y='int.rate',x='fico',data=loans,hue='credit.policy',col='not.fully.paid')
plt.show()

#Transforming the categorical column to numerical values
cat_feats=["purpose"]
final_data=pd.get_dummies(loans,columns=cat_feats,drop_first=True)

#Assigning the data to test and train the model with
X = final_data.drop('not.fully.paid',axis=1)
y = final_data['not.fully.paid']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

#Using a Decision Tree model on the data
dtree=DecisionTreeClassifier()
dtree.fit(X_train,y_train)

#Evaluating the performance
predictions=dtree.predict(X_test)
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))

#Using a Random Forest model on the data
rfc = RandomForestClassifier(n_estimators=500)
rfc.fit(X_train, y_train)

#Evaluating the performance
pred=rfc.predict(X_test)
print(classification_report(y_test,pred))
print(confusion_matrix(y_test,pred))

