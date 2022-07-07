import time
import numpy as np
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras import layers
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class CNNModel(object):
    def __init__(self, input_shape=(29, 1), num_classes=6, model_path=None):
        self.model = keras.Sequential(
            [
                keras.Input(input_shape),
                layers.Conv1D(16, kernel_size=3, activation="relu"),
                layers.MaxPooling1D(pool_size=3),
                layers.Conv1D(32, kernel_size=3, activation="relu"),
                layers.MaxPooling1D(pool_size=3),
                layers.Flatten(),
                layers.Dropout(0.5),
                layers.Dense(32, activation="sigmoid"),
                layers.Dense(num_classes, activation='softmax')
            ]
        )
        self.model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['accuracy'])
        if model_path is not None:
            self.model.load_weights(model_path)

    def predict(self, x):
        preds = self.model.predict(x)
        preds = np.argmax(preds, axis=1)
        return preds

    def fit(self, x, y, model_save_path, batch_size=64, epochs=30):
        history = self.model.fit(x, y, batch_size=batch_size, epochs=epochs, validation_split=0.2,
                                 callbacks=[ModelCheckpoint(filepath=model_save_path, save_weights_only=True,
                                                            monitor='val_accuracy', mode='max', save_best_only=True)])


if __name__ == '__main__':
    model_path = "test.h5"
    sample_size = 20000
    data_x, data_y = np.random.random((sample_size, 29)), np.random.randint(0, 12, size=(sample_size, 1))
    class_num = np.unique(data_y).shape[0]
    data_y = keras.utils.to_categorical(data_y, class_num)
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(data_x, data_y, test_size=0.2)
    model = CNNModel(input_shape=(Xtrain.shape[1], 1), num_classes=class_num)
    model.fit(Xtrain, Ytrain, batch_size=512, epochs=10, model_save_path=model_path)
    model = CNNModel(input_shape=(Xtrain.shape[1], 1), num_classes=class_num, model_path=model_path)
    since = time.time()
    preds = model.predict(Xtest)
    end = time.time()
    print(f'Predict {Xtest.shape[0]} samples in {end - since : .9f}s, {(end - since) / Xtest.shape[0]: .9f}s on avg')