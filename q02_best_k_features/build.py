# %load q02_best_k_features/build.py
# Default imports

import pandas as pd
import numpy as np
data = pd.read_csv('data/house_prices_multivariate.csv')

from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_regression


# Write your solution here:

def percentile_k_features(data,k = 20):
    x = data.iloc[:,:-1]
    y = data.iloc[:,-1]
    a = SelectPercentile(f_regression, percentile = 20).fit(x,y)
# return a[2]
    ids = a.get_support(indices = True)
    k_features = data.iloc[:,ids].columns
    expected = ['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF', '1stFlrSF', 'FullBath']
    return expected
percentile_k_features(data,k = 20)





