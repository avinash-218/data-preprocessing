#importing packages and libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('Data.csv') #import dataset

X = dataset.iloc[:,:-1].values #extract X
Y = dataset.iloc[:,-1].values #extract Y

#handling missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean') #replacing nan by average for numerical values
X[:,1:] = imputer.fit_transform(X[:,1:])

#encoding categorical data
from sklearn.preprocessing import OneHotEncoder
one_hot_encoder = OneHotEncoder()
X = np.column_stack((one_hot_encoder.fit_transform(X[:,0].reshape(-1,1)).toarray(),X[:,1:]))

#label encode Y since it is binary
from sklearn.preprocessing import LabelEncoder
label_encoder=LabelEncoder()
Y = label_encoder.fit_transform(Y)

#train test split
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=1)  #20% test split

#feature scaling using standardization
from sklearn.preprocessing import StandardScaler
standard_scaler = StandardScaler()

X_train[:,3:] = standard_scaler.fit_transform(X_train[:,3:])
#no feature scaling y since they are encoded data
X_test[:,3:] = standard_scaler.transform(X_test[:,3:])
#transform not fit_transform because we can get same transformation and can get some relevant predictions to trainset
