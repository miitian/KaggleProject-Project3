# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 07:08:30 2018

@author: mpagrawa
"""

#import house_11112018 as parent

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sqlalchemy as sql
import statsmodels.formula.api as sm
import scipy.stats as stats

from sklearn import preprocessing

from sqlalchemy import create_engine
engine = create_engine('sqlite://', echo=False)

from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(2)

#%matplotlib inline
sns.set_style('whitegrid')

def StartOverDf():
    house = pd.read_csv('kc_house_data.csv', parse_dates=['date'])
    del house['id']
    del house['lat']
    del house['long']
    return house

def AdjustedRSquare(model,X,Y):
    YHat = model.predict(X)
    n,k = X.shape
    sse = np.sum(np.square(YHat-Y),axis=0)
    sst = np.sum(np.square(Y-np.mean(Y)),axis=0)
    R2 = 1- sse/sst
    adjR2 = R2-(1-R2)*(float(k)/(n-k-1))
    return adjR2

def BackwardElimination(X,y,sl):
    columnList = X.columns.tolist()    
    for i in range(0, len(columnList)):
        regressor_OLS = sm.OLS(y, X[columnList]).fit()
        adjR2_before = regressor_OLS.rsquared_adj        
        maxVar = max(regressor_OLS.pvalues) 
        if maxVar > sl:
            ind = regressor_OLS.pvalues[regressor_OLS.pvalues == max(regressor_OLS.pvalues)].index[0]
            columnList_new = columnList.copy()
            columnList_new.remove(ind)
            temp_OLS = sm.OLS(y, X[columnList_new]).fit()
            adjR2_after = temp_OLS.rsquared_adj
            print('before', adjR2_before)
            print('after', adjR2_after, '\n')
            if adjR2_before > adjR2_after:
                return columnList
            else:
                columnList.remove(ind)    
    return columnList

def PolyFeatureNames(featureNames):
    # interaction features
    featureNames = ['intercept'] + featureNames    
    polyFeatureNames = [];    
    for i,x in enumerate(featureNames):
        for y in featureNames[i:]:
            if (x == 'intercept'):
                polyFeatureNames.append(y)
            elif (x==y):
                polyFeatureNames.append((y+'_Square'))
            else:
                polyFeatureNames.append((x+'_'+y))
    return polyFeatureNames

from sklearn.model_selection import GridSearchCV
#from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import ShuffleSplit
from sklearn.metrics import make_scorer
from sklearn.metrics import r2_score

def performance_metric(y_true, y_predict):
    score = r2_score(y_true, y_predict)
    return score


def fit_model(X,y):
    cv_sets = ShuffleSplit(X.shape[0],n_iter=10,
                           test_size=0.20,
                           random_state=1234)
    ridgeModel = Ridge()
    params = {'alpha':list(range(0,5)),
             'solver' : ('auto', 
                         'svd', 
                         'cholesky', 
                         'lsqr', 
                         'sparse_cg', 
                         'sag', 
                         'saga')}
    scoring_func = make_scorer(performance_metric)
    grid = GridSearchCV(ridgeModel,params,scoring_func,cv=cv_sets)
    grid = grid.fit(X,y)
    return grid.best_estimator_

house = StartOverDf()
#plt.figure(figsize=(15,10))
#sns.heatmap(house.corr(), annot=True, cmap='coolwarm')
max_date = max(house.date)
house['Age_sold'] = house['date'].apply(lambda x: ((max_date - x).days))
house['House_Built_Age'] = 2015-house['yr_built']
house['House_Renovated_Age'] = 2015-house['yr_renovated']
house['Tot_Bathrooms'] = house.bathrooms * house.bedrooms
house['Price_Sqft'] = house.price / house.sqft_living15
house['Price_Sqft_lot'] = house.price / house.sqft_lot15
del(house['bathrooms'])
house.at[house.index[house.bedrooms ==33],'bedrooms'] = 3
house.at[house.index[house.bedrooms ==11],'bedrooms'] = 3


house= pd.get_dummies(house, columns =['view'], drop_first=True)
house= pd.get_dummies(house, columns =['grade'], drop_first=True)
house= pd.get_dummies(house, columns =['zipcode'], drop_first=True)
house= pd.get_dummies(house, columns =['condition'], drop_first=True)
house= pd.get_dummies(house, columns =['floors'], drop_first=True)
house= pd.get_dummies(house, columns =['bedrooms'], drop_first=True)
# sqft_living and lot areas have changed even though house is not renovated
# drop older coloumns considering 15 data as the latest and accurate


X = house.drop(['price','date','sqft_living','sqft_lot','sqft_above','sqft_basement','yr_built','yr_renovated'], axis=1)
y = house.price


# Stats model
X['intercept'] = 1
res = BackwardElimination(X,y,0.05)

#regressor_OLS = sm.OLS(y, X).fit()
#regressor_OLS.summary()

#regressor_OLS.pvalues
#res.remove('intercept')

Xpoly = poly.fit_transform(X[res])
#polyFeatureNames = PolyFeatureNames(X.columns.tolist())
polyFeatureNames = PolyFeatureNames(res)
#polyFeatureNames = PolyFeatureNames(['House_Renovated_Age','tot_bathrooms'])
#Xpoly.shape
XpolyDf = pd.DataFrame(Xpoly, columns=polyFeatureNames)
#X = XpolyDf

#zip_unique = list(set(X.zipcode))

#from statsmodels.stats.multicomp import pairwise_tukeyhsd
#output = pairwise_tukeyhsd(y,X.zipcode)
#output.summary()
#df = pd.DataFrame(output.summary())
#df = pd.DataFrame(data=output._results_table.data[1:], columns=output._results_table.data[0])
#df1 = df[df.reject == False]
#output.plot_simultaneous()[0]



from sklearn.linear_model import LinearRegression
lm = LinearRegression()
from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X[res], y, test_size=0.3, random_state=5)
X_train, X_test, y_train, y_test = train_test_split(X[res], y, test_size=0.3, random_state=5)
lm.fit(X_train, y_train)
coefMetrics = pd.DataFrame(index=X_train.columns, data=lm.coef_)
R2_train = lm.score(X_train, y_train)
R2_test = lm.score(X_test, y_test)
y_pred = lm.predict(X_test)
adjR2_train = AdjustedRSquare(lm,X_train,y_train)
adjR2_test = AdjustedRSquare(lm,X_test,y_test)


from sklearn.linear_model import Lasso
lm1 = Lasso(alpha=1, max_iter=5000)
from sklearn import cross_validation as cv
#X_train_L, X_test_L, y_train_L, y_test_L = cv.train_test_split(X,y, test_size=0.25,random_state=1234)
#X_train, X_test, y_train, y_test = train_test_split(X[res], y, test_size=0.3, random_state=5)
X_train_L, X_test_L, y_train_L, y_test_L = cv.train_test_split(X[res], y, test_size=0.3, random_state=5)
lm1.fit(X_train_L, y_train_L)
coefMetrics = pd.DataFrame(index=X_train_L.columns, data=lm.coef_)
R2_train = lm1.score(X_train_L, y_train_L)
R2_test = lm1.score(X_test_L, y_test_L)
y_pred = lm1.predict(X_test_L)
adjR2_train1 = AdjustedRSquare(lm1,X_train_L,y_train_L)
adjR2_test1 = AdjustedRSquare(lm1,X_test_L,y_test_L)


from sklearn.linear_model import Ridge
lm1 = Ridge(alpha=1,max_iter=5000, solver='svd')
#from sklearn.model_selection import cross_validate as cv
from sklearn import cross_validation as cv
#from sklearn import cross_validation as cv
#X_train_L, X_test_L, y_train_L, y_test_L = cv.train_test_split(X,y, test_size=0.25,random_state=1234)
#X_train, X_test, y_train, y_test = train_test_split(X[res], y, test_size=0.3, random_state=5)
X_train_L, X_test_L, y_train_L, y_test_L = cv.train_test_split(X[res], y, test_size=0.3, random_state=5)
lm1.fit(X_train_L, y_train_L)
coefMetrics = pd.DataFrame(index=X_train_L.columns, data=lm.coef_)
R2_train = lm1.score(X_train_L, y_train_L)
R2_test = lm1.score(X_test_L, y_test_L)
y_pred = lm1.predict(X_test_L)
adjR2_train2 = AdjustedRSquare(lm1,X_train_L,y_train_L)
adjR2_test2 = AdjustedRSquare(lm1,X_test_L,y_test_L)


#best_estimate = fit_model(X_train,y_train)
#best_estimate
#Ridge(alpha=1, copy_X=True, fit_intercept=True, max_iter=None,
#   normalize=False, random_state=None, solver='svd', tol=0.001)
#adjR2_train = 0.9439
#adjR2_test = 0.9419


















