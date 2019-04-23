import os

from keras.models import load_model
import cv2 as cv
import numpy as np
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout

from VieML.ML.resampling import resample
from scipy.optimize import fmin_bfgs
import talos

x = []
y = []

# {'optimizer': ['Adam', 'SGD', 'Adadelta', Adam(lr=0.1), Adam(lr=1), SGD(lr=0.1), SGD(lr=1), Adadelta(lr=0.1),
#                Adadelta(lr=1), Adagrad(lr=0.1), Adagrad(lr=1)],

# p = {'optimizer': ['Adam', 'RMSprop'],
#      'batch_size': [1, 2, 4, 8, 16, 32, 64],
#      'loss': ['mean_squared_error', 'categorical_crossentropy'],
#      'dense_size': [128, 256, 512, 1024, 2048]}

for object_index in range(100):
    image = resample(cv.imread(f"coil-100/obj{object_index + 1}__0.png",
                               cv.IMREAD_GRAYSCALE), 0.25, 0).astype('float32') / 255
    x.append(image.reshape((32, 32, 1)))
    y.append(image.reshape(32 * 32))

size = int(len(x) * 0.8)
x_train = np.array(x[:size])
y_train = np.array(y[:size])
x_test = np.array(x[size:])
y_test = np.array(y[size:])

# x = np.array(x)
# y = np.array(y)


# def conv(x_train, y_train, x_test, y_test, params):
model = Sequential()

model.add(Conv2D(8, (3, 3), input_shape=x_train[0].shape, activation="relu", data_format="channels_last"))
model.add(Conv2D(32, (3, 3), activation="relu"))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation="relu"))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D((2, 2)))
model.add(BatchNormalization())
model.add(Flatten())
# model.add(Dense(params['dense_size'], activation="relu"))
model.add(Dense(1024, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(32 * 32))

# model.compile(optimizer=params['optimizer'], metrics=['mean_absolute_error'], loss=params['loss'])
model.compile(optimizer="Adam", metrics=["mean_absolute_error"], loss="mean_squared_error")
# out = model.fit(x_train, y_train, batch_size=params['batch_size'], epochs=20, validation_data=(x_test, y_test))
model.fit(x_train, y_train, batch_size=2, epochs=200, validation_data=(x_test, y_test))

model.save("multi_angle.h5")

# return out, model


# talos.Scan(x, y, p, conv)

# model = load_model("multi_angle.h5")

print(f"Erreur absolue moyenne : {model.evaluate(x_test, y_test)[1]}")

x_predict = model.predict(x_test)
x_predict = x_predict.reshape((x_predict.shape[0], 32, 32))
y_test = y_test.reshape((y_test.shape[0], 32, 32))

cv.namedWindow("a", cv.WINDOW_NORMAL)
cv.namedWindow("b", cv.WINDOW_NORMAL)
cv.namedWindow("c", cv.WINDOW_NORMAL)
cv.namedWindow("d", cv.WINDOW_NORMAL)
cv.namedWindow("e", cv.WINDOW_NORMAL)
cv.namedWindow("f", cv.WINDOW_NORMAL)
cv.imshow("a", y_test[0])
cv.imshow("b", x_predict[0])
cv.imshow("c", y_test[1])
cv.imshow("d", x_predict[1])
cv.imshow("e", y_test[2])
cv.imshow("f", x_predict[2])

cv.namedWindow("g", cv.WINDOW_NORMAL)
cv.namedWindow("h", cv.WINDOW_NORMAL)
cv.imshow("g", y_train[0].reshape((32, 32)))
cv.imshow("h", model.predict(x_train)[0].reshape((32, 32)))

cv.waitKey(0)
cv.destroyAllWindows()

# model = load_model("multi_angle.h5")
# image = resample(cv.imread("coil-100/obj99__355.png", cv.IMREAD_GRAYSCALE), 0.25, 0).astype('float32') / 255
# image = image.reshape((32, 32, 1))
# image = model.predict(np.array([image]))[0].reshape((32, 32))
# cv.namedWindow('a', cv.WINDOW_NORMAL)
# cv.imshow('a', image)
# cv.waitKey(0)
# cv.destroyAllWindows()
