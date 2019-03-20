from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import SGD
from keras.datasets import mnist
from keras.utils import np_utils
import os
from scipy import misc
from VieML.ML.resampling import resample
from sklearn.model_selection import train_test_split

zoom = 200./1400.
path = ''
high_quality_images = []
low_quality_images = []
for image_name in os.listdir(path):
    complete_path = path + image_name
    image = misc.imread(complete_path)
    high_quality_images.append(image)
    low_quality_images.append(resample(image = image, zoom = zoom, i = 0))

x_train, x_test, y_train, y_test = train_test_split(low_quality_images,high_quality_images,test_size = 0.05)

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