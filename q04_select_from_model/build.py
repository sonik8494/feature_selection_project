# %load q04_select_from_model/build.py
# Default imports
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

data = pd.read_csv('data/house_prices_multivariate.csv')
def select_from_model(data):
    
    np.random.seed(9)
    X = data.iloc[:,:-1]
    y = data.iloc[:,-1]
    a = SelectFromModel(RandomForestClassifier())
    a.fit(X,y)
    b= a.get_support(indices =True).tolist()
    return [X.columns[x] for x in b]

select_from_model(data)


# Your solution code here



