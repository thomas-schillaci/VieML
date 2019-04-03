import QtQuick 2.0


Rectangle {

id : fenetre
width : 500
height : 500



Rectangle {

height : 200
width:250

border.color:"black"
anchors.centerIn:parent

Text  {

text : " Importer l'image ici"
color : "red"
anchors.centerIn :parent

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