# %load q03_rf_rfe/build.py
# Default imports
import pandas as pd

data = pd.read_csv('data/house_prices_multivariate.csv')

from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier


# Your solution code here
def rf_rfe(data):
    
    X = data.iloc[:,:-1]
    y = data.iloc[:,-1]
    a = RFE(RandomForestClassifier())
    a.fit(X,y)
    return X.columns[a.ranking_ == 1].tolist()
rf_rfe(data)



