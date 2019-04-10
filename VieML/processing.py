from keras.models import load_model
from VieML.ML.resampling import resample
from sklearn.preprocessing import MinMaxScaler
import numpy as np

zoom = 1/8.
sc = MinMaxScaler(feature_range = (0,1))

def process(img):
    shape = img.shape
    flatten_img = format(img)
    upscaling_model = load_model('C:/Users/leovu/PycharmProjects/VieML/VieML/ML/Models/upscaling_model.h5')
    multi_angle_model = load_model('C:/Users/leovu/PycharmProjects/VieML/VieML/ML/Models/multi_angle_model.h5')
    flatten_res = multi_angle_model.predict(flatten_img)
    flatten_res = res = upscaling_model.predict(flatten_res)
    res = flatten_res.reshape(int(shape[0]),int(shape[1]))
    res = sc.inverse_transform(res)
    return res

def format(img):
    img = resample(image=img, zoom=zoom, i=0).astype("float32")
    sc.fit_transform(img)
    img = np.ndarray.flatten(img)
    return img