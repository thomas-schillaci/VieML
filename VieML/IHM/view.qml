import QtQuick 2.0

Rectangle {
    width: 500
    height: 500


    Rectangle {

    border.color : "red"
    anchors.centerIn : parent

    Text {
    anchors.centerIn : parent
    text : "Importer Image ici "
    }

    }







    Grid {
     id: colorPicker
     x: 40; y : 450 ; anchors.bottom: page.bottom; anchors.bottomMargin: 4
     rows: 1; columns: 3; spacing: 100

       Button {  cellName : "Import"  }
       Button { cellName : "Save" }
       Button { cellName : "Upgrade" }
}


}