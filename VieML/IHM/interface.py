
from PySide2 import QtCore, QtWidgets, QtGui
import sys


class Fenetre(QtWidgets.QDialog):

    def __init__(self):
        super(Fenetre, self).__init__()
        # Create widgets
        self.button = QtWidgets.QPushButton("import")
        self.button2 = QtWidgets.QPushButton("upgrade")
        self.button3 = QtWidgets.QPushButton("save")
        self.image = QtWidgets.QLabel()
        self.pixmap =QtGui.QPixmap()
        self.image.setPixmap(self.pixmap)
        # Create layout and add widget
        self.button3.hide()
        self.button2.hide()
        self.button.clicked.connect(self.upgrade)
        self.button.clicked.connect(self.explorer)
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.button)
        layout.addWidget(self.button2)
        layout.addWidget(self.button3)
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

if __name__ == '__main__':
    # Create the Qt Application
    app = QtWidgets.QApplication(sys.argv)
    # Create and show the window
    fenetre = Fenetre()
    fenetre.show()
    # Run the main Qt loop
    sys.exit(app.exec_())
