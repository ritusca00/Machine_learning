# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 18:40:09 2019

@author: blizn
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#d:\GITHUB\Machine_learning\01_Data_Preprocessing\
dataset = pd.read_csv("d:\\GITHUB\\Machine_learning\\01_Data_Preprocessing\\Data.csv")
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,3].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""