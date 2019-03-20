import sys
from PySide2.QtWidgets import QApplication
from PySide2.QtQuick import QQuickView
from PySide2.QtCore import QUrl





app = QApplication([])
View = QQuickView()
url = QUrl("View.qml")

View.setSource(url)
View.show()
app.exec_()
