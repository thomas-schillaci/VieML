from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import math
import matplotlib.pyplot as plot

model = Sequential()
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

n = 1000
x = np.array([[k / n * math.pi * 2] for k in range(n)], "float32")
y = np.array([[(math.sin(k / n * math.pi * 2) + 1) / 2] for k in range(n)], "float32")

model.fit(x, y, epochs=1000)

y_predict = model.predict(x)

plot.plot(x, y, x, y_predict)
plot.show()
