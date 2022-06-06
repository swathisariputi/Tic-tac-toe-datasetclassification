#Group No:30
#Problem Statement:Classify tic-tac-toe data set
#Binary classification task on possible configurations of tic-tac-toe game


#python code for classification using Naive Bayes
#Importing the neccesary libraries we are going to need
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
import seaborn as sns
sns.set_style('darkgrid')
df = pd.read_csv('tic-tac-toe.csv')
print(df.shape) #view dimensions of dataset
print(df) #Printing the dataset
le_column = LabelEncoder()
df["top-left-square"] = le_column.fit_transform(df["top-left-square"])                      #changing strings to number format
df["top-middle-square"] = le_column.fit_transform(df["top-middle-square"])
df["top-right-square"] = le_column.fit_transform(df["top-right-square"])
df["middle-left-square"] = le_column.fit_transform(df["middle-left-square"])
df["middle-middle-square"] = le_column.fit_transform(df["middle-middle-square"])
df["middle-right-square"] = le_column.fit_transform(df["middle-right-square"])
df["bottom-left-square"] = le_column.fit_transform(df["bottom-left-square"])
df["bottom-middle-square"] = le_column.fit_transform(df["bottom-middle-square"])
df["bottom-right-square"] = le_column.fit_transform(df["bottom-right-square"])
df["bottom-right-square"] = le_column.fit_transform(df["bottom-right-square"])
#df["class"] = le_column.fit_transform(df["class"])
print(df.head())
#x_df=df.drop('class',axis=1,inplace=True)
#sns.heatmap(x_df)
cor=df.corr()
fig,ax=plt.subplots(figsize=(10,8))
sns.heatmap(cor,cmap='coolwarm',annot=True,fmt=".2f")
plt.xticks(range(len(cor.columns)),cor.columns);
plt.yticks(range(len(cor.columns)),cor.columns)
plt.show()
features = list(df.columns)   #getting all columns in features list
print(features)
features.remove('class')          
X = df[features].values.astype(np.float32) 
y = df.pop("class")
X_train, X_test, Y_train, Y_test = train_test_split(X, y,test_size = 0.5, random_state = 0)
model = GaussianNB()
model.fit(X_train, Y_train)
print("training accuracy :", model.score(X_train, Y_train))
print("testing accuracy :", model.score(X_test, Y_test))
