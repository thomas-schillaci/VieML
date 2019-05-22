import cv2 as cv
import numpy as np
import talos
from keras import Sequential
from keras.engine.saving import load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam, SGD, Adagrad, Adadelta
from sklearn.model_selection import train_test_split
from talos.model import lr_normalizer
from resampling import resample

RES = 128
INPUT_RES = 16
OUTPUT_RES = 64


def get_data():
    global RES, INPUT_RES, OUTPUT_RES

    m_x = []
    m_y = []

    for object_index in range(120):
        for series_index in range(24):
            series = []
            for i in range(3):
                path = f"coil-20/obj{object_index // 6 + 1}__{series_index * 3 + i}.png" if object_index % 6 == 0 else \
                    f"coil-100/obj{object_index - (object_index + 1) // 6 + 1}__{(series_index * 3 + i) * 5}.png"
                image = resample(cv.imread(path, cv.IMREAD_GRAYSCALE), INPUT_RES / RES, 0).astype('float32') / 255
                series.append(image)
                if i == 1:
                    image = resample(cv.imread(path, cv.IMREAD_GRAYSCALE), OUTPUT_RES / RES, 0).astype('float32') / 255
                    m_y.append(image)
            m_x.append(series)

    m_x_train, m_x_test, m_y_train, m_y_test = train_test_split(m_x, m_y, test_size=0.15, shuffle=False)
    m_x_train = np.array(m_x_train)
    m_y_train = np.array(m_y_train)
    m_x_test = np.array(m_x_test)
    m_y_test = np.array(m_y_test)

    m_y_train = m_y_train.reshape((m_y_train.shape[0], OUTPUT_RES * OUTPUT_RES))
    m_y_test = m_y_test.reshape((m_y_test.shape[0], OUTPUT_RES * OUTPUT_RES))

    m_x = np.array(m_x)
    m_y = np.array(m_y)
    m_y = m_y.reshape((m_y.shape[0], OUTPUT_RES * OUTPUT_RES))

    return m_x, m_y, m_x_train, m_x_test, m_y_train, m_y_test


def get_model(m_input_shape, m_params=None):
    global OUTPUT_RES

    if m_params is None:
        m_params = {'input_shape': x[0].shape,
             'optimizer': Adam,
             'lr': 2.2,
             'batch_size': 100,
             'second_layer': 0,
             'dense_size_1': 3000,
             'dense_size_2': 0,
             'dropout': 0.005,
             'dense_activation': 'relu',
             'output_activation': 'linear'}

    m_model = Sequential()

    m_model.add(Conv2D(8, (3, 3), input_shape=m_input_shape, activation="relu", data_format="channels_first"))
    m_model.add(Conv2D(32, (3, 3), activation="relu", dim_ordering="th"))
    m_model.add(MaxPooling2D((2, 2)))
    m_model.add(Conv2D(32, (3, 3), activation="relu", dim_ordering="th"))
    m_model.add(Conv2D(64, (3, 3), activation="relu", dim_ordering="th"))
    m_model.add(MaxPooling2D((2, 2)))
    m_model.add(Flatten())
    m_model.add(Dense(m_params['dense_size_1'], activation=m_params['dense_activation']))
    m_model.add(Dropout(m_params['dropout']))
    if m_params['second_layer'] == 1:
        m_model.add(Dense(m_params['dense_size_2'], activation=m_params['dense_activation']))
        m_model.add(Dropout(m_params['dropout']))
    m_model.add(Dense(OUTPUT_RES * OUTPUT_RES, activation=m_params['output_activation']))

    m_model.compile(m_params['optimizer'](lr=lr_normalizer(m_params['lr'], m_params['optimizer'])),
                    'mean_squared_error', metrics=['mean_absolute_error'])

    return m_model


def conv(m_x_train, m_y_train, m_x_test, m_y_test, m_params):
    m_model = get_model(m_params['input_shape'], m_params)
    m_out = m_model.fit(m_x_train, m_y_train, batch_size=m_params['batch_size'], epochs=3,
                        validation_data=(m_x_test, m_y_test))

    return m_out, m_model


def display(m_model, m_x_test, m_y_test):
    global INPUT_RES, OUTPUT_RES

    print(f"Erreur absolue moyenne : {m_model.evaluate(m_x_test, m_y_test)[1]}")

    m_y_predict = m_model.predict(m_x_test)
    m_y_predict = m_y_predict.reshape((m_y_predict.shape[0], OUTPUT_RES, OUTPUT_RES))

    cv.namedWindow("a", cv.WINDOW_NORMAL)
    cv.namedWindow("b", cv.WINDOW_NORMAL)
    cv.namedWindow("c", cv.WINDOW_NORMAL)
    cv.namedWindow("d", cv.WINDOW_NORMAL)
    cv.namedWindow("e", cv.WINDOW_NORMAL)
    cv.namedWindow("f", cv.WINDOW_NORMAL)
    cv.imshow("a", m_x_test[0][1].reshape((INPUT_RES, INPUT_RES)))
    cv.imshow("b", m_y_predict[0])
    cv.imshow("c", m_x_test[int(m_x_test.shape[0] / 3)][1].reshape((INPUT_RES, INPUT_RES)))
    cv.imshow("d", m_y_predict[int(m_x_test.shape[0] / 3)])
    cv.imshow("e", m_x_test[2 * int(m_x_test.shape[0] / 3)][1].reshape((INPUT_RES, INPUT_RES)))
    cv.imshow("f", m_y_predict[2 * int(m_x_test.shape[0] / 3)])

    cv.waitKey(0)
    cv.destroyAllWindows()


x, y, x_train, x_test, y_train, y_test = get_data()

# model = get_model(x_train[0].shape)
model = load_model("multi_angle.h5")
# model.fit(x_train, y_train, batch_size=2, epochs=5, validation_data=(x_test, y_test))
# model.save("multi_angle.h5")

display(model, x_test, y_test)

# p = {'input_shape': [x[0].shape],
#      'optimizer': [Adam],
#      'lr': [2.2],
#      'batch_size': [100],
#      'second_layer': [0],
#      'dense_size_1': [3000],
#      'dense_size_2': [0],
#      'dropout': [0.005],
#      'dense_activation': ['relu'],
#      'output_activation': ['linear']}
# talos.Scan(x, y, p, conv)
