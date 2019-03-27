import QtQuick 2.0

Rectangle {
    width: 200
    height: 200


    Rectangle {
    color : red
    border.color : "black"
    anchors.centerIn : parent

    Text {
    anchors.centerIn : parent
    text : "Importer Image ici "
    }

    }







    Grid {
     id: colorPicker
     x: 4; anchors.bottom: page.bottom; anchors.bottomMargin: 4
     rows: 1; columns: 3; spacing: 3

    Button {  cellName : "Import"  }
    Button { cellName : "Save" }
    Button { cellName : "Upgrade" }
}


}