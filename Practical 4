# -*- coding: utf-8 -*-

**DATA PREPROCESSING**
"""

import pandas as pd
 from google.colab import files
 uploaded=files.upload()

 df=pd.read_csv('pokemon_data.csv')
 df.head()

df.shape

df.info()

df.describe()

df.isnull()

df.isnull().sum()

"""DEALING WITH MISSING VALUES

 Deleting the missing values
 Imputing the mising values
 Imputing the missing values for cateogorical data

DELETING THE MISSING VALUES
 Delete the whole colum
"""

df2=df.drop(['Type 2'],axis=1)
 df2.isnull().sum()

"""Deleting a single row"""

r=df.drop(labels=[1],axis=0) #or r=df.drop(0)
r.head()
#labels is used to delete a particular row , rows at particular index and a range of rows

"""Deleting particular rows"""

r2=df.drop(labels=[1,15,20],axis=0) #r2.df.drop([0,15,20])
 r2.head(20)

"""Deleting a range of rows"""

#delete a range of rows -index values 10-20
 r3=df.drop(labels=range(40,45),axis=0) #45 excluded  #r3=df.drop(range(10,20))
 r3.head(50)
 #Note that axis is 0 by default

""" IMPUTING THE MISSING VALUE

 FILL THE MISSING WITH ZERO
"""

#replacing the missing values of the any column with 0.
 #replacing the missing value with '0' using 'fillna method
 df['Type 2']=df['Type 2'].fillna(0)
 df['Type 2'].isnull().sum()

""" FILL THE MISSING VALUE WITH MEAN"""

#replcaing with the mean
 df['HP']=df['HP'].fillna(df['HP'].mean())
 #here hp is not mepty but if it would have been it will be filled with it

""" Mean/Median/Mode Imputation:Replcae missing entries with the average(mean),middle value(median),most freuquent(mode)
 REPLACING WITH MEDIAN
"""

df['Attack']=df['Attack'].fillna(df['Attack'].median())

""" REPLACING WITH MODE"""

df['Speed']=df['Speed'].fillna(df['Speed'].mode())

""" DE-DUPLICATE

 De-Duplicate means removing all duplicate values.There is no need for duplicate values in data analysis.These values only effect the
 accuracy and efficiency of the analysis result.To find duplicate values in the dataset ,we will use a simple dataframe function
 i.e,duplicated()
"""

from google.colab import files
 up=files.upload()

d=pd.read_csv('Cardetails.csv')

d.duplicated()

print(d.duplicated().sum())

""" Removing of duplicates
 subset(default=None): Specifies which columns to consider for identifying duplicates.
 If none,all columns are used.
 If a list of column names is provided,duplicates are detected based on those columns.        keep(default='first): Determines which duplicate row to keep.

 'first':keeps the first occurrence and drops the rest.
 'last':keeps the last occurrence and drop others.
 False:Remove all duplicate rows.
 inplace(defualt=False):If True,modifies the dataframe in place and return None.If false,return a new dataframe with duplicates
 removed.
 ignore_index(default=False)-If true,resets the index in the resulting dataframe
"""

d.drop_duplicates(subset=None,keep='first',inplace=True,ignore_index=False)
 #The drop_duplicates method with inplace=False returns a new DataFrame without modifying the original one.

print(d.duplicated().sum())

""" DEALING WITH OUTLIERS (Detection and Removal)

 DETECTION
"""

import sklearn
from sklearn.datasets import load_diabetes
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

diabetes=load_diabetes()
 column_name=diabetes.feature_names
 df_diabetics=pd.DataFrame(diabetes.data,columns=column_name)  #used to create a valid dataset for EDA

df_diabetics.head()

df_diabetics.shape

df_diabetics.info()

#For detection of outliers we used box plot
 sns.boxplot(df_diabetics['bmi'])
 plt.title('Boxplot of BMI')
 plt.show()
 #above the upper line the circles represent the outliers

"""REMOVAL"""

def removal_box_plot(df,column,threshold):
   removed_outliers=df[df[column]<=threshold] # a updated dataset with rows only which value is less than the threshold for
   sns.boxplot(removed_outliers[column])
   plt.title(f"Box Plot without Outliers of {column}")
   plt.show()
   return removed_outliers
 threshold_value=0.12
 no_outliers=removal_box_plot(df_diabetics,'bmi',threshold_value)

"""TRANSFORMATION/SCALING
 Transformation or scaling is a technique used in data preprocessing to change the range or distribution of data. Common scaling
 techniques include:
 Min-Max Scaling: Scales the data to a fixed range, usually between 0 and 1.
 Standardization (Z-score scaling): Transforms the data to have a mean of 0 and a standard deviation of 1
"""

#we do scaling so we can arrange all attribute to a particular range
#z-score=(X-mean)/standard deviation
from scipy import stats
import numpy as np
z=np.abs(stats.zscore(df_diabetics['age'])) #this finds zscore for each cell of age column and takes its absolute(age cant
print(z)
