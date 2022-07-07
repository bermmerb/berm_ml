from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam, SGD
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

df = pd.read_csv('data/user_visit_duration.csv')

print(df.head())
# df.plot(kind='scatter', x='Time (min)', y='Buy')

# model = Sequential()
# model.add(Dense(1, input_shape=(1,), activation='sigmoid'))
# model.compile(SGD(learning_rate=0.5), 'binary_crossentropy', metrics=['accuracy'])
# model.summary()

X = df[['Time (min)']].values
y = df['Buy'].values
# print(X)
# print(y)
# model.fit(X, y, epochs=25)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# # print(X)
# # print(X_train)
train_df = pd.DataFrame(X_train, columns=['X_train'])
train_df['y_train'] = y_train
print(train_df.head())
test_df = pd.DataFrame(X_test, columns=['X_test'])
test_df['y_test'] = y_test
print(test_df.head())
# params = model.get_weights()
# # print(params)
# params = [np.zeros(w.shape) for w in params]
# for w in params:
#    print(np.zeros(w.shape))
# print(params)
# model.set_weights(params)

ax = train_df.plot(kind='scatter', x='X_train', y='y_train',
             title='Purchase behavior VS time spent on site', color='blue', alpha=0.3)
test_df.plot(ax=ax, kind='scatter', x='X_test', y='y_test', color='red', alpha=0.3)


model = Sequential()
model.add(Dense(1, input_shape=(1,), activation='sigmoid'))
model.compile(SGD(learning_rate=0.5), 'binary_crossentropy', metrics=['accuracy'])
model.summary()

model.fit(X_train, y_train, epochs=25)

# y_pred = model.predict(X)
# y_class_pred = y_pred > 0.5
# cm = confusion_matrix(y, y_class_pred)
# print(cm)

# model.fit(X, y, epochs=25)
temp = np.linspace(0, 4)
# print(temp)
# print(X_train)
# print(df[['Time (min)']].values)
# print(model.predict(temp))
ax.plot(temp, model.predict(temp), color='yellow')

temp_class = model.predict(temp) > 0.5

# # # print(temp_class)

ax.plot(temp, temp_class, color='orange')

# y_pred = model.predict(X)
# y_class_pred = y_pred > 0.5

# # print("The accuracy score is {:0.3f}".format(accuracy_score(y, y_class_pred)))

# # params = model.get_weights()

# # plt.legend(['model', 'data', 'predict'])

print("The accuracy score is {:0.3f}".format(accuracy_score(y, model.predict(X) > 0.5)))
print("The train accuracy score is {:0.3f}".format(accuracy_score(y_train, model.predict(X_train) > 0.5)))
print("The test accuracy score is {:0.3f}".format(accuracy_score(y_test, model.predict(X_test) > 0.5)))

plt.show()
