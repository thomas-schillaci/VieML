from scipy import ndimage, misc
import matplotlib.pyplot as plt

test_image = misc.ascent()

def resample(image,zoom,i):
    '''
    Resamples a given image

    :param image:
    :param zoom: size of the resampled image compared with the original
    :param i: Order of the spline interpolation
                0 : 2D-nearest neighbors
                1 : Bilinear
                2 : Bicubic
    :return: A resampled image
    '''
    result = ndimage.zoom(image, zoom=zoom, order = i)
    return result

def plot(image,zoom,i):
    fig = plt.figure()
    ax1 = fig.add_subplot(121)  # left side
    ax2 = fig.add_subplot(122)  # right side
    result = resample(image,zoom,i)
    ax1.imshow(image)
    ax2.imshow(result)
    plt.show()