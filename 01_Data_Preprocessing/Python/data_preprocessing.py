# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#d:\GITHUB\Machine_learning\01_Data_Preprocessing\
dataset = pd.read_csv("d:\\GITHUB\\Machine_learning\\01_Data_Preprocessing\\Data.csv")
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,3].values

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy = 'mean', axis = 0)
imputer.fit(X[:,1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])