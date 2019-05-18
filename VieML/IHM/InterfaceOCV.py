
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
        self.count=0;
        self.adresse=" "
        self.ListeFrame=[]
        self.button1= QtWidgets.QPushButton("Traitement set de donnée")
        self.button2 = QtWidgets.QPushButton("upgrade")
        self.button3 = QtWidgets.QPushButton("save")
        self.button4=QtWidgets.QPushButton("rogner (region fixe)")
        self.button5=QtWidgets.QPushButton("afficher")
        # Create layout and add widget
        self.button4.clicked.connect(self.cropfixe)
        self.button5.clicked.connect(self.afficher)
        self.button3.clicked.connect(self.save)
        self.button1.clicked.connect(self.dataSet)
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.button5)
        layout.addWidget(self.button2)
        layout.addWidget(self.button3)
        layout.addWidget(self.button4)
        layout.addWidget(self.button1)
        layout.maximumSize()

        # Set  layout
        self.setLayout(layout)
        #definition fonctions
    def cropfixe(self): #on definit une zone fixe d'interet( rectangle) et on reconstitue une liste de frame a partir de cette zone d'interet
        self.ListeFrame = []
        if self.adresse == " ":
            self.adresse, _ = QtWidgets.QFileDialog.getOpenFileName(self, caption="open file")
        cap = cv2.VideoCapture(self.adresse)
        i=1
        ret, frame = cap.read();
        r = cv2.selectROI(frame,False)
        imCrop = frame[int(r[1]):int(r[1] + 128), int(r[0]):int(r[0]+128)]
        cv2.imshow('frame', imCrop)
        self.ListeFrame.append(imCrop)
        i = i + 1
        while(i<int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
            ret, frame = cap.read();
            imCrop = frame[int(r[1]):int(r[1] + 128), int(r[0]):int(r[0]+128)]
            cv2.imshow('frame',imCrop)
            cv2.waitKey(0)
            self.ListeFrame.append(imCrop)
            i=i+1
    def upgrade(self):
        self.button3.show()

    def afficher(self):
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self, caption="open file")
        self.adresse=fileName
        cap = cv2.VideoCapture(fileName)
        while (1):
            ret, frame = cap.read()
            cv2.imshow('frame', frame)
            if cv2.waitKey(125) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
        self.button4.show()

    def save(self):
        frame_height, frame_width = (self.ListeFrame[0]).shape[:2]
        out = cv2.VideoWriter('VieML.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 8, (frame_width, frame_height)) #2eme parametre=fps de la video
        for i in range (len(self.ListeFrame)):
            out.write(self.ListeFrame[i])
        out.release()
        cv2.destroyAllWindows()

    def dataSet(self):
        fileName, _ = QtWidgets.QFileDialog.getOpenFileNames(self, caption="open file")
        self.adresse=fileName
        for i in range(len(fileName)):
            self.ListeFrame = []
            cap = cv2.VideoCapture(self.adresse[i])
            j = 1
            ret, frame = cap.read();
            r = cv2.selectROI(frame, False)
            imCrop = frame[int(r[1]):int(r[1] + 128), int(r[0]):int(r[0] + 128)]
            cv2.imshow('frame', imCrop)
            self.ListeFrame.append(imCrop)
            j = j + 1
            while (j < int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
                ret, frame = cap.read();
                imCrop = frame[int(r[1]):int(r[1] + 128), int(r[0]):int(r[0] + 128)]
                self.ListeFrame.append(imCrop)
                j = j + 1
            frame_height, frame_width = (self.ListeFrame[0]).shape[:2]
            out = cv2.VideoWriter('VieML '+str(i)+".avi", cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 8,
                                  (frame_width, frame_height))  # 2eme parametre=fps de la video
            for i in range(len(self.ListeFrame)):
                out.write(self.ListeFrame[i])
            out.release()
            cv2.destroyAllWindows()

if __name__ == '__main__':
    # Create the Qt Application
    app = QtWidgets.QApplication(sys.argv)
    # Create and show the window
    fenetre = Fenetre()
    fenetre.show()
    # Run the main Qt loop
    sys.exit(app.exec_())
