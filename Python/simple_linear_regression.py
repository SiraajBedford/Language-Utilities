#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 20:33:42 2020

@author: siraaj
"""

import sklearn.datasets as ds
import matplotlib.pyplot as plt

X, y = ds.load_boston(return_X_y = True)

# data.feature_names
# Out[31]: 
# array(['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'], dtype='<U7')

X.data

# Import seaborn
import seaborn as sns

# Apply the default theme
sns.set_style()

# Load an example dataset
tips = sns.load_dataset("tips")

# Create a visualization
sns.relplot(
    data=tips,
    x="total_bill", y="tip", col="time",
    hue="smoker", style="smoker", size="size",
)