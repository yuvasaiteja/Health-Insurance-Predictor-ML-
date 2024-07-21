# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 16:04:19 2023

@author: muthi
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

insurance_dataset=pd.read_csv('insurance.csv')

#Brief intro about the dataset
print(insurance_dataset.head())

#Shape of the dataset
print(insurance_dataset.shape)


#Info of the dataset
print(insurance_dataset.info())
























#DATA ANALYSIS

print(insurance_dataset.describe())
sns.set()
plt.figure(figsize=(6,6))
sns.displot(insurance_dataset['age'])
plt.title("AGE DISTRIBUTION")
plt.show()






plt.figure(figsize=(6,6))
sns.countplot(x="sex",data=insurance_dataset)
plt.title("SEX DISTRIBUTION")
plt.show()







plt.figure(figsize=(6,6))
sns.displot(insurance_dataset['bmi'])
plt.title("BMI DISTRIBUTION")
plt.show()









plt.figure(figsize=(6,6))
sns.countplot(x="children",data=insurance_dataset)
plt.title("CHILDREN DISTRIBUTION")
plt.show()







plt.figure(figsize=(6,6))
sns.countplot(x="smoker",data=insurance_dataset)
plt.title("SMOKER DISTRIBUTION")
plt.show()




plt.figure(figsize=(6,6))
sns.countplot(x="region",data=insurance_dataset)
plt.title("REGION DISTRIBUTION")
plt.show()




plt.figure(figsize=(6,6))
sns.displot(insurance_dataset['charges'])
plt.title("CHARGE DISTRIBUTION")
plt.show()















#DATA PRE_PROCESSING



 
insurance_dataset.replace({'sex':{'male':0,'female':1}},inplace=True)


insurance_dataset.replace({'smoker':{'yes':0,'no':1}},inplace=True)


insurance_dataset.replace({'region':{'southeast':0,'southwest':1,'northeast':2,'northwest':3}},inplace=True)


print(insurance_dataset.describe())


print(insurance_dataset)

















#SPLITTING THE FEATURES AND TARGET
X=insurance_dataset.drop(columns='charges',axis=1)
Y=insurance_dataset['charges']


print(X)
print(Y)





#SPLLITING TRAINING AND TESTING DATA 

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)




'''from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)'''


 
#MODEL TRAINING

regressor = LinearRegression()
regressor.fit(X_train, Y_train)









#MODEL EVALUATION
training_data_prediction=regressor.predict(X_train)
r2_train=metrics.r2_score(Y_train, training_data_prediction)
print(r2_train)




#MODEL PREDICTION
test_data_prediction=regressor.predict(X_test)
r2_test=metrics.r2_score(Y_test, test_data_prediction)
print(r2_test)










input_data=[52,1,37.4,0,1,1]
npa=np.asarray(input_data)
npa=npa.reshape(1, -1)
npa=regressor.predict(npa)
print(npa)









'''#RANDOM FOREST REGRESSOR
from sklearn.ensemble import RandomForestRegressor
regressor_rf=RandomForestRegressor()
regressor_rf.fit(X_train,Y_train)



training_data_prediction=regressor_rf.predict(X_train)
r2_train=metrics.r2_score(Y_train, training_data_prediction)
print(r2_train)


test_data_prediction=regressor_rf.predict(X_test)
r2_test=metrics.r2_score(Y_test, test_data_prediction)
print(r2_test)


input_data=[23,0,23.845,0,1,2]
npa=np.asarray(input_data)
npa=npa.reshape(1, -1)

print(regressor_rf.predict(npa))'''







