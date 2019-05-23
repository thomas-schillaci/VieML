import talos
from keras.models import Sequential
from keras.layers import Dense, Dropout
import os
from keras_preprocessing import image as img
import numpy as np
from VieML.ML.resampling import resample
from sklearn.model_selection import train_test_split
import cv2
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
sc1 = MinMaxScaler(feature_range = (0,1))
sc2 = MinMaxScaler(feature_range = (0,1))
zoom1 = 1/4. #128*128
zoom2 = 1/2.
path = 'C:/Users/leovu/Desktop/coil-100/'
high_quality_images = []
low_quality_images = []
for image_name in os.listdir(path):
    complete_path = path + image_name
    image_hq = cv2.imread(complete_path, cv2.IMREAD_GRAYSCALE)
    image_lq = resample(image=image_hq, zoom=zoom1, i=0).astype("float64")
    image_lq = resample(image=image_lq, zoom=1/zoom2, i=0).astype("float64")
    image_hq = resample(image=image_hq, zoom=zoom2, i=0).astype("float64")
    image_lq = sc1.fit_transform(image_lq)
    image_hq = sc2.fit_transform(image_hq)
    image_lq = np.ndarray.flatten(image_lq)
    image_hq = np.ndarray.flatten(image_hq)
    high_quality_images.append(image_hq)
    low_quality_images.append(image_lq)
input_shape = len(low_quality_images[0])
output_shape = len(high_quality_images[0])
high_quality_images = np.array(high_quality_images)
low_quality_images = np.array(low_quality_images)

p = {
    'activation':['relu'],
    'optimizer':['Adam'],
    'losses':['mean_squared_error'],
    'hidden_layers':[2],
    'units':[1000], #<
    'units0':[1000], #<
    'batch_size':[100], #
    'epochs':[100], #
    'kernel_initializer': ["uniform"],
    'dropouts':[0.2] #<
}

def upscale_model(x_train,y_train,x_val,y_val,params):
    classifier = Sequential()
    #Dense Layers
    classifier.add(Dense(units=params['units'], activation=params['activation'],
                    kernel_initializer=params['kernel_initializer'],
                   input_dim=input_shape))
    classifier.add(Dropout(params['dropouts']))
    for i in range(0,params['hidden_layers']):
        classifier.add(Dense(units=params['units0'], activation=params['activation'],
                    kernel_initializer=params['kernel_initializer']))
        classifier.add(Dropout(params['dropouts']))
    #Output
    classifier.add(Dense(output_shape,kernel_initializer=params['kernel_initializer']))
    #Compile
    classifier.compile(optimizer=params['optimizer'], loss=[params['losses']], metrics=['mae'])
    #Training
    out = classifier.fit(x_train, y_train , epochs = params['epochs'] , batch_size = params['batch_size'],validation_data=[x_val,y_val])
    return out, classifier

scan = talos.Scan(low_quality_images,high_quality_images, model = upscale_model,params=p)