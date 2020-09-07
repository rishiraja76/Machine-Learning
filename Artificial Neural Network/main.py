#Importing packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation,Dropout
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report,confusion_matrix
import random

#Reading the file
df = pd.read_csv('lending_club_loan_two.csv')
print(df.info())

#Creating a count plot
sns.countplot(x="loan_status",data=df)
plt.show()

#Creating a histogram
sns.distplot(a=df["loan_amnt"])
plt.show()

#Correlating the data
df_corr=df.corr()
print(df_corr)

#Creating a heat map
sns.heatmap(df_corr)
plt.show()

#Creating a scatter plot
sns.scatterplot(x="installment",y="loan_amnt",data=df)
plt.show()

#Creating a box plot
sns.boxplot(x="loan_status",y="loan_amnt",data=df)
plt.show()

#Printing the summary statistics for the loan amount when grouped by loan status
print(df.groupby("loan_status")["loan_amnt"].describe())

#Printing the unique entries in the column grade and sub grade
print(df["grade"].sort_values().unique())
print(df["sub_grade"].sort_values().unique())

#Creating a count plot
sns.countplot(x="grade",hue="loan_status",data=df)
plt.show()

#Creating a count plot
sns.countplot(x=df["sub_grade"].sort_values(),data=df)
plt.show()

#Creating a count plot
sns.countplot(x=df["sub_grade"].sort_values(),hue="loan_status",data=df)
plt.show()

#Creating a count plot #
f_and_g = df[(df['grade']=='G') | (df['grade']=='F')]
subgrade_order = sorted(f_and_g['sub_grade'].unique())
sns.countplot(x='sub_grade',data=f_and_g,order = subgrade_order,hue='loan_status')

#Converting the categorical values to numerical values and putting it in a new column
df["loan_repaid"]=df["loan_status"].apply(lambda x: 1 if x=="Fully Paid" else 0)
print(df.head())

#Creating a box plot #
df.corr()['loan_repaid'].sort_values().drop('loan_repaid').plot(kind='bar')
plt.show()

#Printing the data
print(df.head())

#Printing the length of the dataframe
print(len(df))

#Creating a series with the number of missing values per column
missing=pd.Series(df.isnull().sum())
print(missing)

#Converting the series to be a percentage of the total dataframe
missing=(missing/396030)*100
print(missing)

#Printing the number of unique job titles
print(df["emp_title"].nunique())
print(df["emp_title"].value_counts())

#Dropping the job title column
df.drop("emp_title",inplace=True,axis=1)

#Creating a count plot
emp_length_order = [ '< 1 year',
                      '1 year',
                     '2 years',
                     '3 years',
                     '4 years',
                     '5 years',
                     '6 years',
                     '7 years',
                     '8 years',
                     '9 years',
                     '10+ years']
sns.countplot(x="emp_length",data=df,order=emp_length_order)
plt.show()

#Creating a count plot
sns.countplot(x="emp_length",hue="loan_status",data=df,order=emp_length_order)
plt.show()

#Visualizing what percent of people per employment category didn't pay back their loan #
emp_co = df[df['loan_status']=="Charged Off"].groupby("emp_length").count()['loan_status']
emp_fp = df[df['loan_status']=="Fully Paid"].groupby("emp_length").count()['loan_status']
emp_len = emp_co/emp_fp
print(emp_len)
emp_len.plot(kind='bar')
plt.show()

#Dropping the emp_length column
df.drop("emp_length",inplace=True,axis=1)

#Creating a series with the number of missing values per column
missing=pd.Series(df.isnull().sum())
print(missing)

#Printing the first 10 entries of the purpose and title columns
print(df['purpose'].head(10))
print(df['title'].head(10))

#Dropping the title column
df.drop("title",inplace=True,axis=1)

#Printing the number of instances for each unique entry
print(df["mort_acc"].value_counts())

#Printing the correlation with the mort_acc column
print(df.corr()["mort_acc"].sort_values())

#Printing the mean of mort_acc column per total_acc #
print(df.groupby("total_acc").mean()["mort_acc"])

#Filling in the missing values with the mean value corresponding to its total_acc value #
total_acc_avg = df.groupby('total_acc').mean()['mort_acc']
print(total_acc_avg[2.0] )
def fill_mort_acc(total_acc, mort_acc):
    if np.isnan(mort_acc):
        return total_acc_avg[total_acc]
    else:
        return mort_acc
df['mort_acc'] = df.apply(lambda x: fill_mort_acc(x['total_acc'], x['mort_acc']), axis=1)

#Removing the missing values
df.dropna(inplace=True)

#Creating a series with the number of missing values per column
missing=pd.Series(df.isnull().sum())
print(missing)

###

#Printing the columns that are non numeric
print(df.select_dtypes(['object']).columns)

#Converting the categorical values to numerical values
df['term'] = df['term'].apply(lambda term: int(term[:3]))

#Dropping the grade column
df = df.drop('grade',axis=1)

#Converting the categorical values to numerical values and then concatinating the coulmns
subgrade_dummies = pd.get_dummies(df['sub_grade'],drop_first=True)
df = pd.concat([df.drop('sub_grade',axis=1),subgrade_dummies],axis=1)

#Converting the categorical values to numerical values and then concatinating the coulmns
dummies = pd.get_dummies(df[['verification_status', 'application_type','initial_list_status','purpose' ]],drop_first=True)
df = df.drop(['verification_status', 'application_type','initial_list_status','purpose'],axis=1)
df = pd.concat([df,dummies],axis=1)

#Converting the categorical values to numerical values and then concatinating the coulmns
print(df['home_ownership'].value_counts())
df['home_ownership']=df['home_ownership'].replace(['NONE', 'ANY'], 'OTHER')
dummies = pd.get_dummies(df['home_ownership'],drop_first=True)
df = df.drop('home_ownership',axis=1)
df = pd.concat([df,dummies],axis=1)

#Extracting the zip code from address and then converting the categorical values to numerical values and then concatinating the coulmns
df['zip_code'] = df['address'].apply(lambda address:address[-5:])
dummies = pd.get_dummies(df['zip_code'],drop_first=True)
df = df.drop(['zip_code','address'],axis=1)
df = pd.concat([df,dummies],axis=1)

#Dropping the issue_d column
df = df.drop('issue_d',axis=1)

#Extracting the year from earliest_cr_line then converting the categorical values to numerical values
df['earliest_cr_year'] = df['earliest_cr_line'].apply(lambda date:int(date[-4:]))
df = df.drop('earliest_cr_line',axis=1)

#Dropping the loan status column
df = df.drop('loan_status',axis=1)

#Assigning the data to be split into training and testing for the model
X = df.drop('loan_repaid',axis=1).values
y = df['loan_repaid'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=101)

#Normalizing the data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Building the model
model = Sequential()
model.add(Dense(78,  activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(39, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(19, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=1,activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam')

#Fitting the model
model.fit(x=X_train,
          y=y_train,
          epochs=25,
          batch_size=256,
          validation_data=(X_test, y_test),
          )

#Saving the model
model.save('full_data_project_model.h5')

#Evaluating performance
losses = pd.DataFrame(model.history.history)
losses[['loss','val_loss']].plot()
predictions = model.predict_classes(X_test)
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))

#Checking an entry against the model
random.seed(101)
random_ind = random.randint(0,len(df))
new_customer = df.drop('loan_repaid',axis=1).iloc[random_ind]
print(model.predict_classes(new_customer.values.reshape(1,78)))
print(df.iloc[random_ind]['loan_repaid'])