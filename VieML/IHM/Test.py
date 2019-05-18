
from PySide2.QtWidgets import QApplication, QPushButton
from PySide2 import QtCore, QtWidgets, QtGui
import sys
from PySide2.QtCore import Slot
import cv2
import numpy as np
##

class Fenetre(QtWidgets.QDialog):

    def __init__(self):
        super(Fenetre, self).__init__()
        # Create widgets
        self.button = QtWidgets.QPushButton("import")
        self.button2 = QtWidgets.QPushButton("upgrade")
        self.button3 = QtWidgets.QPushButton("save")
        self.button4 = QtWidgets.QPushButton("traitement d'un set de données")
        self.image = QtWidgets.QLabel()
        self.pixmap =QtGui.QPixmap()
        self.image.setPixmap(self.pixmap)
        # Create layout and add widget
        self.button3.hide()
        self.button2.hide()
        self.button2.clicked.connect(self.upgrade)
        self.button.clicked.connect(self.explorer)
        self.button4.clicked.connect(self.DataSet)
        layout.addWidget(self.button)
        layout.addWidget(self.button2)
        layout.addWidget(self.button3)
        layout.addWidget(self.button4)
        layout.addWidget(self.image)
        layout.maximumSize()

        # Set  layout
        self.setLayout(layout)
        #definition fonctions
    def upgrade(self):
        self.button3.show()
        self.button2.show()
    def explorer(self):
        fileName,_ = QtWidgets.QFileDialog.getOpenFileName(self, caption="open file")
        self.image.setPixmap(fileName)

        # Read image
        im = cv2.imread(self.image.setPixmap(fileName))
        # Select ROI
        # NO  #r = cv2.selectROI(im)
        fromCenter = False
        r = cv2.selectROI(im, fromCenter)
        # Crop image
        imCrop = im[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]

        # Display cropped image
        cv2.imshow("Image", imCrop)
        cv2.waitKey(0)


if __name__ == '__main__':
    # Create the Qt Application
    app = QtWidgets.QApplication(sys.argv)
    # Create and show the window
    fenetre = Fenetre()
    fenetre.show()
    # Run the main Qt loop


     #   im = cv2.imread(fileName)

##


    sys.exit(app.exec_())

#Greetings
#-m pip install --upgrade pip
##
##

# Greetings
@Slot()
def say_hello():
    print("Button clicked, Hello!")

# Create the Qt Application
app = QApplication(sys.argv)

# Create a button
button = QPushButton("Click me")

#conection
button.clicked.connect(say_hello)

button.show()
app.exec_()