#Importing packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

#Reading the file
customers=pd.read_csv("Ecommerce Customers")

#Printing information about the file
print(customers.head())
print(customers.info())
print(customers.describe())

#Creating a joint plot
sns.jointplot(x="Time on Website",y="Yearly Amount Spent",data=customers)
plt.show()

#Creating a joint plot
sns.jointplot(x="Time on App",y="Yearly Amount Spent",data=customers)
plt.show()

#Creating a joint plot
sns.jointplot(x="Time on App",y="Length of Membership",data=customers,kind="hex")
plt.show()

#Creating a pair plot
sns.pairplot(customers)
plt.show()

#Length of membership seems to be the most correlated feature with Yearly Amount Spent

#Creating a linear relationship plot
sns.lmplot(x="Yearly Amount Spent",y="Length of Membership",data=customers)
plt.show()

#Assigning the data to test and train the model with
X=customers[["Avg. Session Length","Time on App","Time on Website","Length of Membership"]]
y=customers["Yearly Amount Spent"]
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.3,random_state=101)

#Using a Linear Regression model on the data
lm=LinearRegression()
lm.fit(X_train,y_train)
print('Coefficients: \n{}'.format(lm.coef_))

#Evaluating the performance
predictions=lm.predict(X_test)
print("MAE: {}".format(metrics.mean_absolute_error(predictions,y_test)))
print("MSE: {}".format(metrics.mean_squared_error(predictions,y_test)))
print("RMSE: {}".format(np.sqrt(metrics.mean_squared_error(predictions,y_test))))

#Creating a pair plot
sns.scatterplot(y_test,predictions)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
plt.show()

#Creating a histogram
sns.distplot((predictions-y_test))
plt.show()

#Printing the coeffecients
coeffs=pd.DataFrame(data=lm.coef_,index=["Avg. Session Length","Time on App","Time on Website","Length of Membership"],columns=["Coeffecient"])
print(coeffs)

#Length of membership matters the most for profit
#The company should care more about the mobile app as it provides more profit over the web app