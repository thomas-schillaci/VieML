import QtQuick 2.2
import QtQuick.Dialogs 1.0


Rectangle {

id : fenetre
width : 500
height : 500


FileDialog {
id: fileDialog
title: "Choose a file"
selectExisting: true



}





Rectangle {

height : 200
width:250

border.color:"black"
anchors.centerIn:parent

Item  {

anchors.centerIn :parent

Button { cellName:" importer l'image ici"; cellColor:"white" ; width : 150
anchors.centerIn:parent


MouseArea {
anchors.fill:parent
onClicked :console.log("Please")

}
}

}

}

Grid {
 x : 40
 y: 450

 columns : 3
 spacing : 50

 Button {cellName :"importer" ; cellColor :"grey" }
 Button {cellName:"sauvergarder" ; cellColor : "grey"}
 Button {cellName :"upgrade"; cellColor : "grey" }

}

}


