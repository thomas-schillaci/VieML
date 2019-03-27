import QtQuick 2.0

    Item {
    id : container
    property alias cellName : buttonText.text
    anchors.horizontalCenter: page.horizontalCenter
    width: 20; height: 15

    Rectangle {

    id: rectangle
    border.color : "white"
    anchors.fill : parent
    anchors.centerIn: parent
    color : "red"




    Text {
    id : buttonText
    anchors.centerIn: parent
    font.pointSize: 10; font.bold: true


    }

    }

    }









