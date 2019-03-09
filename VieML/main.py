from keras.models import Sequential
from keras.layers import Dense
import numpy as np


def test():
    model = Sequential()
    model.add(Dense(16, activation='relu', input_dim=2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['binary_accuracy'])

    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], "float32")
    y = np.array([[0], [0], [0], [1]], "float32")

    model.fit(x, y, epochs=200)

    y_predict = model.predict(x).round()
    print(y_predict)


test()
