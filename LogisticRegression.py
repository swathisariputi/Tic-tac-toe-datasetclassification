#Group No:30
#Problem Statement:Classify tic-tac-toe dataset
#Binary classification task on possible configurations of tic-tac-toe game

#imported necessary libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

#definition of class LogisticRegression:
class LogisticRegressionFromScratch:
    def __init__(self, learning_rate, n_iters):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit_S(self, X, y):
        n_samples, n_features = X.shape     #n_samples=no of instances in X , n_features=no of attributes 
        # init parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        # gradient descent
        for _ in range(self.n_iters):
            # approximate y with linear combination of weights and x, plus bias
            linear_model = np.dot(X, self.weights) + self.bias     #wx+b
            # apply sigmoid function
            y_predicted = self._sigmoid_S(linear_model)

            # compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted-y)) #X.T=X transpose T
            db = (1 / n_samples) * np.sum(y_predicted-y)        
            # update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict_S(self, X):
        linear_model = np.dot(X, self.weights) + self.bias  #b+WX
        y_predicted = self._sigmoid_S(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]        #class 1-->positive-->X wins game else O
        return np.array(y_predicted_cls)

    #return sigmoid value
    def _sigmoid_S(self, x):
        return 1 / (1 + np.exp(-x))

    def accuracy_S(self,y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)   
        return accuracy

# Testing
if __name__ == "__main__":

    df = pd.read_csv('tic-tac-toe.csv')
    print("Given raw data is :")
    print(df)

    #decodes categorial values to numeric values
    le_obj = LabelEncoder()
    df["top-left-square"] = le_obj.fit_transform(df["top-left-square"])
   
    df["top-middle-square"] = le_obj.fit_transform(df["top-middle-square"])

    df["top-right-square"] = le_obj.fit_transform(df["top-right-square"])

    df["middle-left-square"] = le_obj.fit_transform(df["middle-left-square"])

    df["middle-middle-square"] = le_obj.fit_transform(df["middle-middle-square"])

    df["middle-right-square"] = le_obj.fit_transform(df["middle-right-square"])

    df["bottom-left-square"] = le_obj.fit_transform(df["bottom-left-square"])

    df["bottom-middle-square"] = le_obj.fit_transform(df["bottom-middle-square"])

    df["bottom-right-square"] = le_obj.fit_transform(df["bottom-right-square"])

    df["class"]  =  le_obj.fit_transform(df["class"])

    print("\nData After tranformation is :")
    print(df)
    
    #creating the list with df.coloumns
    attributes = list(df.columns)
    print("\nAttributes in the dataset are:")
    print(attributes)
    print("\n")
    attributes.remove('class')
        
    #creating input and oupt sets X and y 
    X = df[attributes].values.astype(np.float32) #input dataset
    y = df.pop("class")         #output dataset

    #spliting the dataset to tet and train data 
    #Setting random_state a fixed value will guarantee that same sequence of text and train datasets will be generated everytime we run the code
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

    #creating object for class LogisticRegressionFromScratch by passing alpha as 0.4 and no of iterations as 1000
    regressor = LogisticRegressionFromScratch(learning_rate=0.4, n_iters=1000)

    #calling fit_S method to train the data
    regressor.fit_S(X_train, y_train)

    #calling predict_S method and testing the data
    predictions1 = regressor.predict_S(X_test)

    #printing accuracy 
    print("LogisticRegresion From Scratch classification Testdata accuracy:", regressor.accuracy_S(y_test, predictions1))
    
    #imported LogisticRegression from sklearn and creating model
    model = LogisticRegression()
    model.fit(X_train, y_train)
    predictions2 = model.predict(X_test)
    
    accuracy=accuracy_score(y_test,predictions2)
    
    print("LogistticRegression Inbuilt  classification Testdata accuracy  :", accuracy)

    #create a correlation matrix
    sns.set_style('darkgrid')   
    cor=df.corr()                                           #create an object of corr()
    fig,ax=plt.subplots(figsize=(10,8))                     #sets plot size
    sns.heatmap(cor,cmap='coolwarm',annot=True,fmt=".2f")   #create a heatmap annote=true displays values in grid 
    plt.xticks(range(len(cor.columns)),cor.columns)         #apply xticks
    plt.yticks(range(len(cor.columns)),cor.columns)         #apply yticks
    plt.show()                                              #show the plot

    