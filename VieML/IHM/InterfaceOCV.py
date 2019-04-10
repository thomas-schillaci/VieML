
from PySide2 import QtCore, QtWidgets, QtGui
import sys
import numpy as np
import cv2
import argparse
import re
class Fenetre(QtWidgets.QDialog):

    def __init__(self):
        super(Fenetre, self).__init__()
        # Create widgets
        self.adresse=" "
        self.button2 = QtWidgets.QPushButton("upgrade")
        self.button3 = QtWidgets.QPushButton("save")
        self.button4=QtWidgets.QPushButton("rogner")
        self.button5=QtWidgets.QPushButton("afficher")
        # Create layout and add widget
        self.button4.clicked.connect(self.crop)
        self.button5.clicked.connect(self.afficher)
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.button5)
        layout.addWidget(self.button2)
        layout.addWidget(self.button3)
        layout.addWidget(self.button4)
        layout.maximumSize()

        # Set  layout
        self.setLayout(layout)
        #definition fonctions
    def crop(self):
        if self.adresse==" ":
            self.adresse, _ = QtWidgets.QFileDialog.getOpenFileName(self, caption="open file")




    def upgrade(self):
        self.button3.show()

    def afficher(self):
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self, caption="open file")
        cap = cv2.VideoCapture(fileName)
        while (1):
            ret, frame = cap.read()
            cv2.imshow('frame', frame)
            if cv2.waitKey(500) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
        self.button4.show()
        self.adresse=fileName

if __name__ == '__main__':
    # Create the Qt Application
    app = QtWidgets.QApplication(sys.argv)
    # Create and show the window
    fenetre = Fenetre()
    fenetre.show()
    # Run the main Qt loop
    sys.exit(app.exec_())
