import QtQuick 2.0


Item {
 id:container
 height :40
 width : 100
 property alias cellColor : rectangle.color
 property alias cellName : buttonText.text



 Rectangle {
 id:rectangle
 anchors.fill:parent
 border.color:"black"





  Text {
  id: buttonText

  anchors.centerIn:parent


  }

  }





}






