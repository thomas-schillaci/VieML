from keras.models import load_model
import cv2 as cv
import numpy as np
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split

from VieML.ML.resampling import resample
import talos

x = []
y = []

# {'optimizer': ['Adam', 'SGD', 'Adadelta', Adam(lr=0.1), Adam(lr=1), SGD(lr=0.1), SGD(lr=1), Adadelta(lr=0.1),
#                Adadelta(lr=1), Adagrad(lr=0.1), Adagrad(lr=1)],

# p = {'optimizer': ['Adam', 'RMSprop'],
#      'batch_size': [1, 2, 4, 8, 16, 32, 64],
#      'loss': ['mean_squared_error', 'categorical_crossentropy'],
#      'dense_size': [128, 256, 512, 1024, 2048]}

START_RES = 16
END_RES = 64

for object_index in range(100):
    for series_index in range(24):
        series = []
        for i in range(3):
            path = f"coil-100/obj{object_index + 1}__{(series_index * 3 + i) * 5}.png"
            image = resample(cv.imread(path, cv.IMREAD_GRAYSCALE), START_RES / 128, 0).astype('float32') / 255
            series.append(image)
            if i == 1:
                image = resample(cv.imread(path, cv.IMREAD_GRAYSCALE), END_RES / 128, 0).astype('float32') / 255
                y.append(image)
        x.append(series)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.05, shuffle=False)
x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

y_train = y_train.reshape((y_train.shape[0], END_RES * END_RES))
y_test = y_test.reshape((y_test.shape[0], END_RES * END_RES))

# x = np.array(x)
# y = np.array(y)

# def conv(x_train, y_train, x_test, y_test, params):
model = Sequential()

model.add(Conv2D(8, (3, 3), input_shape=x_train[0].shape, activation="relu", data_format="channels_first"))
model.add(Conv2D(32, (3, 3), activation="relu", dim_ordering="th"))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation="relu", dim_ordering="th"))
model.add(Conv2D(64, (3, 3), activation="relu", dim_ordering="th"))
model.add(MaxPooling2D((2, 2)))
# model.add(BatchNormalization())
model.add(Flatten())
# model.add(Dense(params['dense_size'], activation="relu"))
# model.add(Dense(4096, activation="relu"))
# model.add(Dropout(0.2))
model.add(Dense(2048, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(END_RES * END_RES))

model.compile('Adam', 'mean_squared_error', metrics=['mean_absolute_error'])

model.fit(x_train, y_train, batch_size=32, epochs=50, validation_data=(x_test, y_test))

model.save("multi_angle.h5")

# model.compile(optimizer=params['optimizer'], metrics=['mean_absolute_error'], loss=params['loss'])
# model.compile(optimizer='Adam', metrics=["mean_absolute_error"], loss="mean_squared_error")

# out = model.fit(x_train, y_train, batch_size=params['batch_size'], epochs=20, validation_data=(x_test, y_test))
# model.fit(x_train, y_train, batch_size=2, epochs=200, validation_data=(x_test, y_test))

# model.save("multi_angle2.h5")

# return out, model


# talos.Scan(x, y, p, conv)

model = load_model("multi_angle.h5")

print(f"Erreur absolue moyenne : {model.evaluate(x_test, y_test)[1]}")

y_predict = model.predict(x_test)
y_predict = y_predict.reshape((y_predict.shape[0], END_RES, END_RES))
y_test = y_test.reshape((y_test.shape[0], END_RES, END_RES))

cv.namedWindow("a", cv.WINDOW_NORMAL)
cv.namedWindow("b", cv.WINDOW_NORMAL)
cv.namedWindow("c", cv.WINDOW_NORMAL)
cv.namedWindow("d", cv.WINDOW_NORMAL)
cv.namedWindow("e", cv.WINDOW_NORMAL)
cv.namedWindow("f", cv.WINDOW_NORMAL)
cv.imshow("a", x_test[0][1].reshape((START_RES, START_RES)))
cv.imshow("b", y_predict[0])
cv.imshow("c", x_test[int(x_test.shape[0] / 3)][1].reshape((START_RES, START_RES)))
cv.imshow("d", y_predict[int(x_test.shape[0] / 3)])
cv.imshow("e", x_test[2 * int(x_test.shape[0] / 3)][1].reshape((START_RES, START_RES)))
cv.imshow("f", y_predict[2 * int(x_test.shape[0] / 3)])

cv.waitKey(0)
cv.destroyAllWindows()
