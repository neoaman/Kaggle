# Classification Problem.

"""
KMEANS Classification 
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Data Import
df = pd.read_csv("train.csv")
df.head()
Test = pd.read_csv("test.csv")
# Imputation 
df_1 = df.loc[df['Survived']==1]
df_1.Age.mean()
df_1['Age'].replace(np.nan,28,inplace=True)

df_0 = df.loc[df['Survived']==0]
df_0.Age.mean()
df_0['Age'].replace(np.nan,30,inplace=True)

# Concating rows
df_n = pd.concat([df_0,df_1])
df_n.isna().sum()
df_n.columns

# Imputation in categorical 
df_n['Embarked'].mode()
df_n['Embarked'].replace(np.nan,"S",inplace=True)
df_n.columns

# Target and independent 
x_train =  df_n[['Pclass', 'Sex', 'Age', 'SibSp',
       'Parch', 'Fare', 'Embarked']]
x_train.head()
y_train = df_n['Survived']
y_train.head()

# Changing Categorical to numeric

# Run once
x_train['Sex'].replace("male",1,inplace=True) # Male as 1
x_train['Sex'].replace("female",0,inplace=True) # Female as 0

x_train['Sex'].replace("male",1,inplace=True) # Male as 1
x_train['Sex'].replace("female",0,inplace=True) # Female as 0
x_train.Embarked.value_counts()
x_train['Embarked'].replace("S",0,inplace=True) # S as 0
x_train['Embarked'].replace("C",1,inplace=True) # C as 1
x_train['Embarked'].replace("Q",2,inplace=True) # Q as 2


# Imputationn in Test data
Test.isna().sum()
Test['Age'].mean()
Test['Fare'].mean()

# Run Once
Test['Age'].replace(np.nan,30,inplace=True)
Test['Fare'].replace(np.nan,36,inplace=True)
Test.isna().sum()

# test data
x_test =  Test[['Pclass', 'Sex', 'Age', 'SibSp',
       'Parch', 'Fare', 'Embarked']]
y_test = pd.read_csv("gender_submission.csv")['Survived']

# Categorical to numeric conversion in test data
x_test['Sex'].replace("male",1,inplace=True) # Male as 1
x_test['Sex'].replace("female",0,inplace=True) # Female as 0

x_test['Sex'].replace("male",1,inplace=True) # Male as 1
x_test['Sex'].replace("female",0,inplace=True) # Female as 0
x_test.Embarked.value_counts()
x_test['Embarked'].replace("S",0,inplace=True) # S as 0
x_test['Embarked'].replace("C",1,inplace=True) # C as 1
x_test['Embarked'].replace("Q",2,inplace=True) # Q as 2


# Model Building 1st model KNN
        modelKnn = KNeighborsClassifier(n_neighbors=8).fit(x_train,y_train)
        y_knn = modelKnn.predict(x_test)
        
        # Accuracy
        accuracy_score(y_test,y_knn) # Getting 65% Accuracy in KNN
        
    # Standardised data
    x = StandardScaler().fit(x_train).transform(x_train.astype(float))
    x_stest = StandardScaler().fit(x_test).transform(x_test.astype(float))
        modelKnn2 = KNeighborsClassifier(n_neighbors=8).fit(x,y_train)
        y_knn2 = modelKnn2.predict(x_stest)
        accuracy_score(y_test,y_knn2) # Getting 85% Accuracy in KNN

# Model Building 2nd Model with Decission Tree
        modelTree =  DecisionTreeClassifier(criterion="entropy",random_state=3,max_depth=3).fit(x_train,y_train)       
        y_tree = modelTree.predict(x_test)        
        accuracy_score(y_test,y_tree) # Getting 88.9% Accuracy in Decission tree

        modelTree2 =  DecisionTreeClassifier(criterion="gini",random_state=3,max_depth=3).fit(x_train,y_train)       
        y_tree2 = modelTree2.predict(x_test)        
        accuracy_score(y_test,y_tree2) # Getting 89.4% Accuracy in Decission tree

# Model using Rendom Forest
        modelRF = RandomForestClassifier(n_jobs=5,oob_score=True,n_estimators=100,criterion="entropy",max_depth=3).fit(x_train,y_train)
        y_RF = modelRF.predict(x_test)
        accuracy_score(y_test,y_RF) # Getting 87.5% Accuracy in Decission tree
        
        modelRF = RandomForestClassifier(n_jobs=5,oob_score=True,n_estimators=100,criterion="entropy",max_depth=3,random_state=6).fit(x,y_train)
        y_RF = modelRF.predict(x_stest)
        accuracy_score(y_test,y_RF) # Getting 95.45% Accuracy in Decission tree
        

        modelRF2 = RandomForestClassifier(n_jobs=5,oob_score=True,n_estimators=100,criterion="gini",max_depth=3).fit(x,y_train)
        y_RF2 = modelRF2.predict(x_stest)
        accuracy_score(y_test,y_RF2) # Getting 94.0% Accuracy in Decission tree
a = []
        for i in range(1,30) :
            modelRF = RandomForestClassifier(n_jobs=5,oob_score=True,n_estimators=100,criterion="gini",max_depth=3,random_state=i).fit(x,y_train)
            y_RF = modelRF.predict(x_stest)
            a.append(accuracy_score(y_test,y_RF))
            
a

Survival = pd.DataFrame(y_RF)
PassengerId=Test[["PassengerId"]]

A = pd.concat([PassengerId,Survival],axis=1)
A.columns = ["PassengerId","Survival"]
type(A)

A.to_csv("AMAN.csv")
