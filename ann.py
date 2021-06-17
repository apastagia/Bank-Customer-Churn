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

#data splitting
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#make artificial neural network
#import libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout #to avoid the overfitting(regularization technique)

#initialize ANN
classifier = Sequential()

#adding first ip layer and hidden layer
classifier.add(Dense(units=10, kernel_initializer='he_normal', activation='relu', input_dim=11))
#adding second hidden layer
classifier.add(Dense(units=15, kernel_initializer='he_normal', activation='relu'))
#adding third hidden layer
classifier.add(Dense(units=20, kernel_initializer='he_normal', activation='relu'))
#adding output layer
classifier.add(Dense(units=1, kernel_initializer='glorot_uniform', activation='sigmoid'))
print('')
print(classifier.summary())
print('')

#compile ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#fitting ANN to training set
model_history = classifier.fit(X_train, y_train, validation_split=0.33, batch_size=10, epochs=100)
#list all data in history
print(model_history.history.keys())

#summarized history for accuracy
plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'test'], loc = 'upper left')
plt.show()

#test set result
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
score = accuracy_score(y_test, y_pred)