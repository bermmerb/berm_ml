from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam, SGD
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def line(x, w=0, b=0):
   return x * w + b

def mean_squared_error(y_true, y_pred):
   s = (y_true - y_pred)**2
   return s.mean()


df = pd.read_csv('data/weight-height.csv')

X = df[['Height']].values
y_true = df['Weight'].values

X_train, X_test, y_train, y_test = train_test_split(X, y_true, test_size=0.2)

model = Sequential()
model.add(Dense(1, input_shape=(1,)))
model.summary()
model.compile(Adam(learning_rate=0.8), 'mean_squared_error')

model.fit(X_train, y_train, epochs=40)

y_pred = model.predict(X)
W, B = model.get_weights()
print(f'Linear finction: y = {W}x + {B}')
print("The R2 score is {:0.3f}".format(r2_score(y_true, y_pred)))

df.plot(kind='scatter',
        x='Height',
        y='Weight',
        title='Weight and Height in adults',
        alpha=0.3)
plt.plot(X, y_pred, color='red')

plt.show()

y_train_pred = model.predict(X_train).ravel()
y_test_pred = model.predict(X_test).ravel()

print("The Mean Squared Error on the Train set is:\t{:0.1f}".format(mse(y_train, y_train_pred)))
print("The Mean Squared Error on the Test set is:\t{:0.1f}".format(mse(y_test, y_test_pred)))
print("The R2 score on the Train set is:\t{:0.3f}".format(r2_score(y_train, y_train_pred)))
print("The R2 score on the Test set is:\t{:0.3f}".format(r2_score(y_test, y_test_pred)))



