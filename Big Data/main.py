#Needs to run on Ubuntu with Spark and Hadoop installed or on AWS

#Importing the package
from pyspark import SparkContext

#Creating an instance
sc = SparkContext()

#Reading the file
services = sc.textFile('services.txt')

#Printing first 2 entries
print(services.take(2))

#Formatting the data
cleanServ = services.map(lambda x: x[1:] if x[0]=='#' else x).map(lambda x: x.split())

#Printing the data
print(cleanServ.collect())

#Printing 4th and last column
print(cleanServ.map(lambda lst: (lst[3],lst[-1])).collect())

#Grouping and adding the amounts
finalServ = cleanServ.map(lambda lst: (lst[3],lst[-1])).reduceByKey(lambda amt1,amt2 : float(amt1)+float(amt2)).filter(lambda x: not x[0]=='State').sortBy(lambda stateAmount: stateAmount[1], ascending=False)

#Printing the data
print(finalServ.collect())

