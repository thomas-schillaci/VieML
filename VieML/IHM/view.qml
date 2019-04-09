import QtQuick 2.2
import QtQuick.Dialogs 1.0
import QtQuick.Layouts 1.1
import QtQuick.Window 2.0
import QtQuick 2.12
import QtQuick.Controls 1.4







Rectangle {



id : fenetre
width : 750
height : 750


FileDialog {
id: fileDialog
title: "Choose a file"
selectExisting: true
selectMultiple: false
        selectFolder: false





}

Grid {
x:40
y:350
rows:2
spacing:50

Button {  text:"importer"

  MouseArea {
        anchors.fill: parent
        onClicked: { fileDialog.open()}
    }
 }
Button2{cellName :"rogner" ; cellColor :"grey" }
}








Rectangle {

height : 200
width:250

border.color:"black"
anchors.centerIn:parent

Item  {

anchors.centerIn :parent

Button2 { cellName:" importer l'image ici"; cellColor:"white" ; width : 150
anchors.centerIn:parent



}

}

}

Grid {
 x : 40
 y: 650

 columns : 3
 spacing : 100

 Button2 {cellName :"zommer" ; cellColor :"grey" }
 Button2 {cellName:"sauvergarder" ; cellColor : "grey"}
 Button2 {cellName :"upgrade"; cellColor : "grey" }

}

}


