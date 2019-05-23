from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adadelta
from keras.datasets import mnist
from keras.utils import np_utils
import os
from keras_preprocessing import image as img
import numpy as np
from VieML.ML.resampling import resample
from sklearn.model_selection import train_test_split
import cv2 as cv
import talos
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
#sc1 = MinMaxScaler(feature_range = (0,1))
#sc2 = MinMaxScaler(feature_range = (0,1))


zoom1 = 1/2. #128*128 #1/2
zoom2 = 1
path = 'C:/Users/leovu/PycharmProjects/VieML/VieML/coil-100/'
high_quality_images = []
low_quality_images = []
for image_name in os.listdir(path):
    complete_path = path + image_name
    '''
    image_hq = cv2.imread(complete_path,cv2.IMREAD_GRAYSCALE)
    image_lq = resample(image = image_hq, zoom = zoom1, i=0).astype("float64")
    image_lq = resample(image=image_lq, zoom=1/zoom2, i=0).astype("float64")
    image_hq = resample(image = image_hq, zoom = zoom2, i = 0).astype("float64")
    image_lq = sc1.fit_transform(image_lq)
    image_hq = sc2.fit_transform(image_hq)
    #image_lq = np.expand_dims(image_lq, axis=0)
    '''
    image_lq = resample(cv.imread(complete_path, cv.IMREAD_GRAYSCALE), zoom1, 0).astype('float32') / 255
    image_hq = resample(cv.imread(complete_path, cv.IMREAD_GRAYSCALE), zoom2, 0).astype('float32') / 255
    image_lq = np.ndarray.flatten(image_lq)
    image_hq = np.ndarray.flatten(image_hq)
    high_quality_images.append(image_hq)
    low_quality_images.append(image_lq)


x_train, x_test, y_train, y_test = train_test_split(low_quality_images,high_quality_images,test_size = 0.01)

x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

#input_shape = low_quality_images[0].shape
input_shape = len(low_quality_images[0])
output_shape = len(high_quality_images[0])
model = Sequential()
'''
model.add(Conv2D(64, (4, 4), strides= 1,input_shape=input_shape, activation="relu", data_format="channels_first"))
model.add(MaxPooling2D(pool_size = (2,2), dim_ordering = "th"))
model.add(Conv2D(64, (4, 4), strides = 1, activation="relu"))
model.add(MaxPooling2D(pool_size = (2,2), dim_ordering = "th"))
model.add(Flatten())
'''
model.add(Dense(units = 1500, activation='relu', #5000
                kernel_initializer='uniform'
                ,input_dim=input_shape))
#model.add(Dense(8000, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(1500, activation="relu",kernel_initializer='uniform')) #4000
model.add(Dropout(0.2))
model.add(Dense(1500, activation="relu",kernel_initializer='uniform'))
model.add(Dropout(0.2))
model.add(Dense(1500, activation="relu",kernel_initializer='uniform'))
model.add(Dropout(0.2))
model.add(Dense(output_shape ,kernel_initializer='uniform'))

model.compile(loss="mse", optimizer='adam', metrics=['mae'])

model.fit(x_train, y_train, batch_size=100, epochs=400, validation_data=(x_test, y_test))

model.save('C:/Users/leovu/PycharmProjects/VieML/VieML/ML/Models/upscaling_model.h5')

from keras.models import load_model
model = load_model('C:/Users/leovu/PycharmProjects/VieML/VieML/ML/Models/upscaling_model.h5')
'''
liste = model.predict(x_test)

img = y_test[0].reshape((int(128*zoom2),int(128*zoom2)))
#img = sc2.inverse_transform(img)
plt.imshow(img)
plt.show()

img = x_test[0].reshape((int(128*zoom2),int(128*zoom2)))
#img = sc1.inverse_transform(img)
plt.imshow(img)
plt.show()

img = liste[0].reshape((int(128*zoom2),int(128*zoom2)))
#img = sc2.inverse_transform(img)
plt.imshow(img)
plt.show()
'''
#pas assez d'importance sur les détails au centre de l'image, et puis pour des images qui auraient un fond non noir ça pourrait poser problème