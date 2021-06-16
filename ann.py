import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('Churn Modeling.csv')

X = df.iloc[:, 3:13]
y = df.iloc[:, 13]

geo = pd.get_dummies(X['Geography'], drop_first=True)
gender = pd.get_dummies(X['Gender'], drop_first=True)

X = pd.concat([X, geo, gender], axis = 1)
X = X.drop(['Geography', 'Gender'], axis = 1)