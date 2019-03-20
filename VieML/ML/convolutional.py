from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import SGD
from keras.datasets import mnist
from keras.utils import np_utils

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], 1, 28, 28).astype('float32') / 255
x_test = x_test.reshape(x_test.shape[0], 1, 28, 28).astype('float32') / 255
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

model = Sequential()

model.add(Conv2D(32, (5, 5), input_shape=(1, 28, 28), activation="relu", data_format="channels_first"))
model.add(MaxPooling2D())
model.add(Conv2D(64, (5, 5), activation="relu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(1000, activation="relu"))
model.add(Dense(10, activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer=SGD(lr=0.01), metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=200, epochs=10, validation_data=(x_test, y_test))

print(model.evaluate(x_test, y_test))
