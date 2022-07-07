from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam, SGD
# from keras.wrappers.scikit_learn import KerasClassifier
from scikeras.wrappers import KerasClassifier
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import cross_val_score, KFold
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def build_logistic_regression_model():
   model = Sequential()
   model.add(Dense(1, input_shape=(1,), activation='sigmoid'))
   model.compile(SGD(learning_rate=0.5),
                 'binary_crossentropy',
                 metrics=['accuracy'])
   return model

df = pd.read_csv('data/user_visit_duration.csv')

print(df.head())

X = df[['Time (min)']].values
y = df['Buy'].values

model = KerasClassifier(model=build_logistic_regression_model,
                        epochs=25,
                        verbose=0)

cv = KFold(3, shuffle=True)

scores = cross_val_score(model, X, y, cv=cv)
print(scores)
print("The cross validation accuracy is {:0.4f} ± {:0.4f}".format(scores.mean(), scores.std())) 