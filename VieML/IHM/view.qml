import QtQuick 2.2
import QtQuick.Dialogs 1.0
import QtQuick.Layouts 1.1
import QtQuick.Window 2.0
import QtQuick 2.12
import QtQuick.Controls 1.4
import QtMultimedia 5.12







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
onFileUrlChanged: {

text1.text=fileDialog.fileUrl
mediaPlayer.source=fileDialog.fileUrl
}





}

Grid {
x:40
y:350
rows:3
spacing:50

Button {  text:"importer"

  MouseArea {
        anchors.fill: parent
        onClicked: { fileDialog.open()}
    }
 }
Button2{cellName :"rogner" ; cellColor :"grey" }

Rectangle {
height:40
width:200
color:"red"

Text {
id:text1
 text:"path of the image selected : "
}
}
}










Rectangle {

x:375
y:200
height : 200
width:250

border.color:"black"




MediaPlayer {
        id: mediaPlayer
        autoPlay: true
        autoLoad: true
        loops:MediaPlayer.Infinite

    }

    VideoOutput {
        id:videoOutput
        source:mediaPlayer
        anchors.fill: parent
    }



/*Button2 {
id: importer_demande
cellName:" importer l'image ici"; cellColor:"white" ; width : 150
anchors.centerIn:parent



}
*/



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


