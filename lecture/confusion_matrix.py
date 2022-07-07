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

def pretty_confusion_matrix(y_true, y_pred, labels=["False", "True"]):
   cm = confusion_matrix(y_true, y_pred)
   pred_labels = ['Predicted '+ l for l in labels]
   df = pd.DataFrame(cm, index=labels, columns=pred_labels)
   return df

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

df = pd.read_csv('data/user_visit_duration.csv')

print(df.head())

X = df[['Time (min)']].values
y = df['Buy'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

train_df = pd.DataFrame(X_train, columns=['X_train'])
train_df['y_train'] = y_train
print(train_df.head())
test_df = pd.DataFrame(X_test, columns=['X_test'])
test_df['y_test'] = y_test
print(test_df.head())

model = Sequential()
model.add(Dense(1, input_shape=(1,), activation='sigmoid'))
model.compile(SGD(learning_rate=0.5), 'binary_crossentropy', metrics=['accuracy'])
model.summary()

model.fit(X_train, y_train, epochs=25)

y_pred = model.predict(X)
y_class_pred = y_pred > 0.5
# print(y)
# print(y_class_pred)
print(confusion_matrix(y, y_class_pred))
print(pretty_confusion_matrix(y, y_class_pred, ['Not Buy', 'Buy']))

print("Precision:\t{:0.3f}".format(precision_score(y, y_class_pred)))
print("Recall:  \t{:0.3f}".format(recall_score(y, y_class_pred)))
print("F1 Score:\t{:0.3f}".format(f1_score(y, y_class_pred)))

print(classification_report(y, y_class_pred))