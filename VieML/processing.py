from keras.models import load_model
from VieML.ML.resampling import resample
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
import cv2 as cv
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


zoom = 1/8.
sc = MinMaxScaler(feature_range = (0,1))

def process(img_list):
    '''
        Given a list of images of the same shape, the algorithm extracts and upscales a picture of interest

        :param list of images:
        :return: image
        '''

    shape = img_list[0].shape
    #format_images = []
    #for img in img_list:
    #    format_img = format(img)
    #    format_images.append(format_img)
    upscaling_model = load_model('C:/Users/leovu/PycharmProjects/VieML/VieML/ML/Models/upscaling_model.h5')
    multi_angle_model = load_model('C:/Users/leovu/PycharmProjects/VieML/VieML/ML/Models/multi_angle.h5')
    flatten_res = multi_angle_model.predict(img_list)
    results = flatten_res.reshape((flatten_res.shape[0], OUTPUT_RES, OUTPUT_RES))
    res = resample(image=results[0],zoom=2,i=0)
    flatten_res = upscaling_model.predict(res)
    res = flatten_res.reshape(int(shape[0]),int(shape[1]))
    #res = sc.inverse_transform(res)
    return res

def format(img):
    img = resample(image=img, zoom=zoom, i=0).astype("float32")
    #sc.fit_transform(img)
    img = np.ndarray.flatten(img)
    return img

x, y, x_train, x_test, y_train, y_test = get_data()
res = process(x_test)
