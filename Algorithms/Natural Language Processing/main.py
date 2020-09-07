#Importing packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline

#Reading the file
yelp=pd.read_csv("yelp.csv")

#Printing information about the file
print(yelp.head())
print(yelp.info())
print(yelp.describe())

#Adding a new column
yelp["text length"]=yelp["text"].apply(len)

#Creating a histogram
g = sns.FacetGrid(yelp,col="stars")
g = g.map(plt.hist,"text length")
plt.show()

#Creating a box plot
sns.boxplot(y="text length",x="stars",data=yelp)
plt.show()

#Creating a count plot
sns.countplot(x="stars",data=yelp)
plt.show()

#Grouping and correlating
gb=yelp.groupby(by="stars").mean()
corr=gb.corr()

#Creating a heat map
sns.heatmap(corr)
plt.show()

#Creating a dataframe with only 1 and 5 stars
yelp_class=yelp[(yelp["stars"]==1) | (yelp["stars"]==5)]

#Assigning the data to test and train the model with
x=yelp_class["text"]
y=yelp_class["stars"]

#Scaling the data
cv=CountVectorizer()
x=cv.fit_transform(x) 

#Splitting the data to train and test
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.3,random_state=101)

#Using a Multinomial model on the data
nb=MultinomialNB()
nb.fit(x_train,y_train)

#Evaluating the performance
predictions=nb.predict(x_test)
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))

#Creating a pipeline for the data to follow
pipeline=Pipeline([
    ("CV",CountVectorizer()),
    ("TFIDF",TfidfTransformer()),
    ("NB",MultinomialNB())
])

#Assigning the data to test and train the model with
X=yelp_class["text"]
Y=yelp_class["stars"]
X_train, X_test, Y_train, Y_test=train_test_split(X,Y,test_size=0.3,random_state=101)

#Using the pipeline on the data
pipeline.fit(X_train,Y_train)

#Evaluating the performance
pred=pipeline.predict(X_test)
print(confusion_matrix(Y_test,pred))
print(classification_report(Y_test,pred))
