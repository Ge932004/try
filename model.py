import subprocess
# subprocess.call([r'new_env\Scripts\activate.bat'], shell=True)

import sklearn
import numpy as np

import itertools
import joblib
import pandas as pd
import xgboost
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import StackingClassifier

csv_path = r'Twotypes_smote.csv'
df = pd.read_csv(csv_path)
X = df.drop(['Progression','Gender'],axis = 1)
Y = df['Progression']
base_classifiers = {
    'xgb': xgboost.XGBClassifier(n_estimators = 50),
    'dt': DecisionTreeClassifier(max_depth = None)
} 
    
clf = StackingClassifier(estimators=list(base_classifiers.items()), final_estimator=MLPClassifier())
clf.fit(X, Y)

joblib.dump(clf, "clfnew.pkl")
