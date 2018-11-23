# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 14:25:05 2018

@author: mpagrawa
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sqlalchemy as sql

from sklearn import preprocessing

from sqlalchemy import create_engine
engine = create_engine('sqlite://', echo=False)

#%matplotlib inline
sns.set_style('whitegrid')

def AdjustedRSquare(model,X,Y):
    YHat = model.predict(X)
    n,k = X.shape
    sse = np.sum(np.square(YHat-Y),axis=0)
    sst = np.sum(np.square(Y-np.mean(Y)),axis=0)
    R2 = 1- sse/sst
    adjR2 = R2-(1-R2)*(float(k)/(n-k-1))
    return adjR2

def CreateAndEvaluateModel(model,X,Y):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5)
    model.fit(X_train, y_train)
    R2_train = model.score(X_train, y_train)
    R2_test = model.score(X_test, y_test)
    y_pred = model.predict(X_test)
    from sklearn.metrics import r2_score
    r2_score(y_test, y_pred)
    adjR2_train = AdjustedRSquare(model,X_train,y_train)
    adjR2_test = AdjustedRSquare(model,X_test,y_test)
    print("R2_train= ", R2_train)
    print("R2_test= ", R2_test)
    print("adjR2_train= ", adjR2_train)
    print("adjR2_test= ", adjR2_test)    
    return R2_train,R2_test,adjR2_train, adjR2_test

def StartOverDf():
    house = pd.read_csv('kc_house_data.csv', parse_dates=['date'])
    del house['id']
    del house['lat']
    del house['long']
    return house


#EDA
house = StartOverDf()
max_date = max(house.date)
house['Age_sold'] = house['date'].apply(lambda x: ((max_date - x).days))
house['tot_bathrooms'] = house.bathrooms * house.bedrooms
plt.figure(figsize=(15,10))
sns.heatmap(house.corr(), annot=True, cmap='coolwarm')



resultSet = pd.DataFrame(columns=['Approach', 'Model','R2_train','AdjR2_tain','R2_test','AdjR2_test'])


#1st approach
        #base model (LinearRegression)
house = StartOverDf()
from sklearn.linear_model import LinearRegression
model = LinearRegression()
X = house.drop(['price','date'], axis=1)
y = house.price
result = list(CreateAndEvaluateModel(model, X, y))
result.insert(0,1)
result.insert(1,model)
resultSet.loc[len(resultSet)] = result

#2nd approach
        #base model (LinearRegression)
house = StartOverDf()
from sklearn.linear_model import LinearRegression
model = LinearRegression()
house['bathrooms'] = house.bathrooms * house.bedrooms
max_date = max(house.date)
house['Age_sold'] = house['date'].apply(lambda x: ((max_date - x).days))
X = house.drop(['price','date','bathrooms'], axis=1)
y = house.price
result = list(CreateAndEvaluateModel(model, X, y))
result.insert(0,2)
result.insert(1,model)
resultSet.loc[len(resultSet)] = result

#3rd approach
        #base model (LinearRegression)
        #handle bedrooms outlier
house = StartOverDf()
from sklearn.linear_model import LinearRegression
model = LinearRegression()
house['bathrooms'] = house.bathrooms * house.bedrooms
max_date = max(house.date)
house['Age_sold'] = house['date'].apply(lambda x: ((max_date - x).days))
house.to_sql('house_sql', con=engine)
#house[[house['zipcode'] == 98103 & house['grade']==7]]

# average bedrooms are 2.818 ~ 3
engine.execute('''SELECT avg(bedrooms) FROM house_sql where --bedrooms in (3,33) 
--and zipcode = 98103 
grade= 7
and floors = 1
and (sqft_living > 1600 and sqft_living < 1650)
and (sqft_above > 1000 and sqft_above < 1050)
and bedrooms != 33
''').fetchall()

#house.loc[house.index[house.bedrooms ==33]].bedrooms = 3
house.at[house.index[house.bedrooms ==33],'bedrooms'] = 3

#house.bedrooms[house.bedrooms ==33] = 3

engine.execute('''SELECT bedrooms FROM house_sql where --bedrooms in (1,11) 
--zipcode = 98106 
grade= 7
and floors = 2
and (sqft_living > 2800 and sqft_living < 3100)
and (sqft_above > 2300 and sqft_above < 2500)''').fetchall()

house.at[house.index[house.bedrooms ==11],'bedrooms'] = 3

X = house.drop(['price','date','bathrooms'], axis=1)
y = house.price
result = list(CreateAndEvaluateModel(model, X, y))
result.insert(0,3)
result.insert(1,model)
resultSet.loc[len(resultSet)] = result


#4th approach
        #base model (LinearRegression)
        #Standardize Age
house = StartOverDf()
from sklearn.linear_model import LinearRegression
model = LinearRegression()
house['bathrooms'] = house.bathrooms * house.bedrooms
max_date = max(house.date)
house['Age_sold'] = house['date'].apply(lambda x: ((max_date - x).days))
from sklearn.preprocessing import StandardScaler
standardScaler = StandardScaler()
age = standardScaler.fit_transform(house['Age_sold'].values.reshape(-1,1))
house['Age_sold'] = age
X = house.drop(['price','date'], axis=1)
y = house.price
result = list(CreateAndEvaluateModel(model, X, y))
result.insert(0,4)
result.insert(1,model)
resultSet.loc[len(resultSet)] = result


#5th approach
        #base model (LinearRegression)
        #handle bedroom outliers
house = StartOverDf()
from sklearn.linear_model import LinearRegression
model = LinearRegression()
house['bathrooms'] = house.bathrooms * house.bedrooms
max_date = max(house.date)
house['Age_sold'] = house['date'].apply(lambda x: ((max_date - x).days))
from sklearn.preprocessing import StandardScaler
standardScaler = StandardScaler()
house.at[house.index[house.bedrooms ==33],'bedrooms'] = 3
house.at[house.index[house.bedrooms ==11],'bedrooms'] = 3
age = standardScaler.fit_transform(house['Age_sold'].values.reshape(-1,1))
house['Age_sold'] = age
X = house.drop(['price','date'], axis=1)
y = house.price
result = list(CreateAndEvaluateModel(model, X, y))
result.insert(0,5)
result.insert(1,model)
resultSet.loc[len(resultSet)] = result






