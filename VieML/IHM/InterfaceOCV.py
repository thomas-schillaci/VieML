
from PySide2 import QtCore, QtWidgets, QtGui
import sys
import numpy as np
import cv2
import argparse
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PySide2.QtGui import *
from PySide2.QtCore import *
from PySide2.QtWidgets import *
import os
from math import *
class Fenetre(QtWidgets.QDialog):

   def __init__(self):
       super(Fenetre, self).__init__()
       # Create widgets
       self.count=0;
       self.adresse=" "
       self.ListeFrame=[]
       self.button1= QtWidgets.QPushButton( QtGui.QIcon("Database.png"),"Dataset preparation")
       self.button2 = QtWidgets.QPushButton(QtGui.QIcon("upgrade.png"),"upgrade")
       self.button3 = QtWidgets.QPushButton("save")
       self.button4 = QtWidgets.QPushButton(QtGui.QIcon("index.png"),"Dataset preparation (video)")
       self.button5=QtWidgets.QPushButton("afficher")
       self.Compteur=QtWidgets.QLabel("")
       self.Compteur.setAlignment(Qt.AlignCenter)
       # Create layout and add widget
       self.button4.clicked.connect(self.dataSetVideo)
       self.button5.clicked.connect(self.afficher)
       self.button3.clicked.connect(self.save)
       self.button1.clicked.connect(self.dataSet)
       #Design

       #layout
       layout = QtWidgets.QVBoxLayout()
       label = QLabel()
       pixmap = QPixmap('IA.jpg')
       label.setPixmap(pixmap)
       label.setAlignment(Qt.AlignCenter)

       #layout.addWidget(self.button5)
       layout.addWidget(label)
       layout.addWidget(self.button1)
       layout.addWidget(self.button2)
       #layout.addWidget(self.button3)
       layout.addWidget(self.button4)
       layout.addWidget(self.Compteur)

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
       imCrop = frame[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
       cropped=cv2.resize(imCrop,(128,128))
       cv2.imshow('frame', cropped)
       self.ListeFrame.append(cropped)
       i = i + 1
       while(i<int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
           ret, frame = cap.read();
           imCrop = frame[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
           cropped = cv2.resize(imCrop, (128, 128))
           cv2.imshow('frame', cropped)
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
       Dossier = QtWidgets.QFileDialog.getExistingDirectoryUrl(caption="SELECTIONNER DOSSIER DE DESTINATIION").path()[1:]
       out = cv2.VideoWriter(Dossier+'/VieML.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 8, (frame_width, frame_height)) #2eme parametre=fps de la video
       for i in range (len(self.ListeFrame)):
           out.write(self.ListeFrame[i])
       out.release()
       cv2.destroyAllWindows()

   def dataSetVideo(self):
       Dossier = QtWidgets.QFileDialog.getExistingDirectoryUrl(caption="SELECTIONNER DOSSIER DE DESTINATIION").path()[1:]
       fileName, _ = QtWidgets.QFileDialog.getOpenFileNames(self, caption="open file")
       self.adresse=fileName
       for i in range(len(fileName)):
           self.ListeFrame = []
           cap = cv2.VideoCapture(self.adresse[i])
           j = 1
           ret, frame = cap.read();
           r = cv2.selectROI(frame, False)
           imCrop = frame[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
           cropped = cv2.resize(imCrop, (128, 128))
           self.ListeFrame.append(cropped)
           j = j + 1
           while (j < int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
               ret, frame = cap.read();
               if ret:
                   imCrop = frame[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
                   cropped = cv2.resize(imCrop, (128, 128))
                   self.ListeFrame.append(cropped)
               j = j + 1
           print(Dossier)
           frame_height, frame_width = (self.ListeFrame[0]).shape[:2]
           out = cv2.VideoWriter(Dossier+'/VieML '+str(i)+".avi", cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 20,(frame_width, frame_height))  # 2eme parametre=fps de la video
           for i in range(len(self.ListeFrame)):
               out.write(self.ListeFrame[i])
           out.release()
       cv2.destroyAllWindows()

   def dataSet(self):
       Dossier = QtWidgets.QFileDialog.getExistingDirectoryUrl(caption="Selectionner dossier de destination").path()[1:]
       fileName, _ = QtWidgets.QFileDialog.getOpenFileNames(self, caption="open file")
       self.adresse=fileName
       AncienFichier=os.listdir(Dossier)
       print(AncienFichier)
       if(AncienFichier==[]):
           DerniereUtilisation=0
       else:
           index=0
           DerniereUtilisation=1
           test=True
           eliminerzeros=True
           while(test):
               try:
                   if(eliminerzeros):
                       if AncienFichier[-1][index]!="0":
                           DerniereUtilisation = DerniereUtilisation + int(AncienFichier[-1][index]) * int(pow(10, index))
                           eliminerzeros=False
                       index=index+1
                   else:
                       DerniereUtilisation=DerniereUtilisation+int(AncienFichier[-1][index])*int(pow(10,2-index))
                       index = index + 1
               except ValueError or OverflowError:
                   test=False
                   break
       for i in range(len(fileName)):
           self.Compteur.setText(str(i)+"/"+str(len(fileName)))
           self.ListeFrame = []
           cap = cv2.VideoCapture(self.adresse[i])
           j = 0
           ret, frame = cap.read();
           r = cv2.selectROI(frame, False)
           imCrop = frame[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
           cropped = cv2.resize(imCrop, (128, 128))
           if (DerniereUtilisation+i<10):
               cv2.imwrite(Dossier + '/00' + str(DerniereUtilisation+i) + "_" + "0" + ".jpg", cropped)
           if(DerniereUtilisation+i>9 and DerniereUtilisation+i<100):
               cv2.imwrite(Dossier + '/0' + str(DerniereUtilisation + i) + "_" + "0" + ".jpg", cropped)
           if(DerniereUtilisation+i>99):
               cv2.imwrite(Dossier + '/' + str(DerniereUtilisation + i) + "_" + "0" + ".jpg", cropped)
           j = j + 1
           while (j < int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
               ret, frame = cap.read();
               if ret:
                   imCrop = frame[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
                   cropped = cv2.resize(imCrop, (128, 128))
                   if (DerniereUtilisation + i < 10):
                       cv2.imwrite(Dossier + '/00' + str(DerniereUtilisation + i) + "_" + str(j) + ".jpg", cropped)
                   if (DerniereUtilisation + i > 9 and DerniereUtilisation + i < 100):
                       cv2.imwrite(Dossier + '/0' + str(DerniereUtilisation + i) + "_" + str(j) + ".jpg", cropped)
                   if (DerniereUtilisation + i > 99):
                       cv2.imwrite(Dossier + '/' + str(DerniereUtilisation + i) + "_" + str(j) + ".jpg", cropped)
               j = j + 1
       self.Compteur.setText("")
       cv2.destroyAllWindows()
if __name__ == '__main__':
   # Create the Qt Application
   app = QtWidgets.QApplication(sys.argv)
   # Create and show the window
   fenetre = Fenetre()
   fenetre.show()
   # Run the main Qt loop
   sys.exit(app.exec_())

