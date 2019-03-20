import QtQuick 2.0

Item {
    id: container
    property alias CellColor: rectangle.color
    property alias CellName = buttonText.text

    Rectangle {

    id: rectangle
    anchors.fill : parent



    Text {
    id : buttonText
    anchors.centerIn: parent
    font.pointSize: 24; font.bold: true

    }

    }

    }








