#Importing packages
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix,classification_report

#Reading the file and formatting it to be used in a model
college=pd.read_csv("College_Data")
college.set_index("Unnamed: 0",inplace=True)
college.index.name=None

#Printing the information about the file
print(college.head())
print(college.info())
print(college.describe())

#Creating a scatter plot
sns.scatterplot(x="Grad.Rate",y="Room.Board",hue="Private",data=college)
plt.show()

#Creating a scatter plot
sns.scatterplot(x="F.Undergrad",y="Outstate",hue="Private",data=college)
plt.show()

#Creating a histogram
college[college["Private"]=="Yes"]["Outstate"].plot.hist()
college[college["Private"]=="No"]["Outstate"].plot.hist()
plt.xlabel("Outstate")
plt.show()

#Creating a histogram
college[college["Private"]=="Yes"]["Grad.Rate"].plot.hist()
college[college["Private"]=="No"]["Grad.Rate"].plot.hist()
plt.xlabel("Grad Rate")
plt.show()

#Finding and editing the outlier in the data
print(college[college["Grad.Rate"]>100])
college['Grad.Rate']['Cazenovia College'] = 100

#Creating a histogram
college[college["Private"]=="Yes"]["Grad.Rate"].plot.hist()
college[college["Private"]=="No"]["Grad.Rate"].plot.hist()
plt.xlabel("Grad Rate")
plt.show()

#Using a K Means model on the data
kmeans=KMeans(n_clusters=2)
kmeans.fit(college.drop("Private",axis=1))
print(kmeans.cluster_centers_)

#Converting the categorical column to numerical values
def priv(x):
    if x == "Yes'":
        return 1
    else:
        return 0
college["Cluster"]=college["Private"].apply(priv)

#Evaluating the performance of the model
print(confusion_matrix(college["Cluster"],kmeans.labels_))
print(classification_report(college["Cluster"],kmeans.labels_,))
