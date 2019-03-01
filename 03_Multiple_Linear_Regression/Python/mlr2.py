# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 17:20:41 2019

@author: blizn
"""



import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#d:\GITHUB\Machine_learning\03_Multiple_Linear_Regression\50_Startups.csv
dataset = pd.read_csv("d:\\GITHUB\\Machine_learning\\03_Multiple_Linear_Regression\\50_Startups.csv")
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,4].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,3] = labelencoder_X.fit_transform(X[:,3])
onehotencoder = OneHotEncoder(categorical_features = [3]) #index
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap
X = X[:,1:]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

#BACKWARD ELIMINATION
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)
X_opt = X[:, [0,1,2,3,4,5]]


import statsmodels.formula.api as sm
def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x
 
SL = 0.05
X_Modeled = backwardElimination(X_opt, SL)


# Splitting the dataset into the Training set and Test set
X_opt_train, X_opt_test, y_opt_train, y_opt_test = train_test_split(X_Modeled, y, test_size = 1/3, random_state = 0)
regressor_opt = LinearRegression()
X_opt_train_new = X_opt_train[:,1].reshape(-1,1)
X_opt_test_new = X_opt_test[:,1].reshape(-1,1)
regressor_opt.fit(X_opt_train_new, y_opt_train)
 
y_opt_pred = regressor_opt.predict(X_opt_test.reshape(-1,1))

plt.scatter(X_opt_train_new, y_opt_train, color = 'red')
plt.plot(X_opt_train_new, regressor_opt.predict(X_opt_train_new), color = 'green')
plt.show()

plt.scatter(X_opt_test_new, y_opt_test, color = 'red')
plt.plot(X_opt_train_new, regressor_opt.predict(X_opt_train_new), color = 'green')
plt.show()


