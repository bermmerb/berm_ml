from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam, SGD
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import KFold, cross_val_score

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def pretty_confusion_matrix(y_true, y_pred, labels=["False", "True"]):
   cm = confusion_matrix(y_true, y_pred)
   pred_labels = ['Predicted '+ l for l in labels]
   df = pd.DataFrame(cm, index=labels, columns=pred_labels)
   return df

def build_logistic_regression_model():
   model = Sequential()
   model.add(Dense(1, input_dim=20, activation='sigmoid'))
   model.compile(Adam(learning_rate=0.5), 'binary_crossentropy', metrics=['accuracy'])
   return model

df = pd.read_csv('data/HR_comma_sep.csv')
# print(df.head())
# print(df.info())
# print(df.describe())

print(df.left.value_counts() / len(df))

# df['average_montly_hours'].plot(kind='hist')

df['average_montly_hours_100'] = df['average_montly_hours']/100.0
# df['average_montly_hours_100'].plot(kind='hist')
# plt.show()

# df['time_spend_company'].plot(kind='hist')
# plt.show()

df_dummies = pd.get_dummies(df[['sales', 'salary']])
# print(df_dummies.head())

X = pd.concat([df[['satisfaction_level', 'last_evaluation', 'number_project',
                   'time_spend_company', 'Work_accident',
                   'promotion_last_5years', 'average_montly_hours_100']],
               df_dummies], axis=1).values
y = df['left'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = Sequential()
model.add(Dense(1, input_dim=20, activation='sigmoid'))
model.compile(Adam(learning_rate=0.5), 'binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10)

y_test_pred = (model.predict(X_test) > 0.5).astype("int32")
# print(y_test_pred)
# y_test_pred = np.argmax(model.predict(X_test), axis=-1)

print(pretty_confusion_matrix(y_test, y_test_pred, labels=['Stay', 'Leave']))
print(classification_report(y_test, y_test_pred))

model = KerasClassifier(model=build_logistic_regression_model,
                        epochs=10)

cv = KFold(5, shuffle=True)
scores = cross_val_score(model, X, y, cv=cv)

print("The cross validation accuracy is {:0.4f} ± {:0.4f}".format(scores.mean(), scores.std()))