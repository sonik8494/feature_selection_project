# %load q05_forward_selected/build.py
# Default imports
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split as tts
import numpy as np
from sklearn.metrics import mean_squared_error

data = pd.read_csv('data/house_prices_multivariate.csv')

model = LinearRegression()

def forward_selected(data,model):
    old_r2_score = 0
    new_r2_score = 1
    features = list(data.drop('SalePrice',axis=1).columns)
    selected_features = []
    r2_score_features = []
    X_selected = pd.DataFrame()
    result = pd.DataFrame()
    y = data['SalePrice']
    while(True):
        scores = []
        for i in range(len(features)):
            X = data[features[i]]
            X_selected = result
            X_selected = pd.concat([X_selected,X], axis=1)
            model.fit(X_selected,y)
            y_pred = model.predict(X_selected)
            scores.append(r2_score(y,y_pred))
            X_selected = result
            np_scores = np.array(scores)
        new_r2_score = np_scores.max()
        if(new_r2_score>old_r2_score):
            old_r2_score=new_r2_score
            result = pd.concat([result,data[features[np.argmax(np_scores)]]], axis=1)
            data = data.drop(features[np.argmax(np_scores)],axis = 1)
            selected_features.append(features[np.argmax(np_scores)])
            r2_score_features.append(new_r2_score)
            features.remove(features[np.argmax(np_scores)])
        else:
            break
    return selected_features,r2_score_features
forward_selected(data,model)




