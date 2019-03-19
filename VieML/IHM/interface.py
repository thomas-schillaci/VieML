from PySide2.QtWidgets import (QLineEdit, QPushButton, QApplication,
    QVBoxLayout, QDialog, QFrame)
from PySide2.QtCore import SLOT
import sys


class Fenetre(QDialog):

    def __init__(self):
        super(Fenetre, self).__init__()
        # Create widgets
        self.button = QPushButton("import")
        self.button2= QPushButton("upgrade")
        self.button3=QPushButton("save")
        self.image=QFrame()
        # Create layout and add widgets
        layout = QVBoxLayout()
        layout.addWidget(self.button)
        layout.addWidget(self.button2)
        layout.addWidget(self.button3)
        layout.addWidget(self.image)
        # Set dialog layout
        self.setLayout(layout)





if __name__ == '__main__':
    # Create the Qt Application
    app = QApplication(sys.argv)
    # Create and show the form
    fenetre = Fenetre()
    fenetre.show()
    # Run the main Qt loop
    sys.exit(app.exec_())
