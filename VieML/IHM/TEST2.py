# Convert hu into normalized values between 0 and 255
img = ( (img - img.max())/(img.max()-img.min()) ) * -1
img *= 255
img = img.astype(int)
img = (255 - img)

# Convert to opencv format
a = np.expand_dims(img, axis = 2)
img = np.concatenate((a, a, a), axis = 2)
img = np.require(img, np.uint8, 'C')

# QT Stuff
width, height, channel = img.shape
bytesPerLine = 3 * width
imgQT = QImage(img, height, width, bytesPerLine,
               QImage.Format_RGB888).rgbSwapped()
self.imgQP = QPixmap.fromImage(imgQT)
imgQPrs = self.imgQP.scaled(768, 768)
self.scene_edit.addPixmap(imgQPrs)
self.edit_l.setScene(self.scene_edit)