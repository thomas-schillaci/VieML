import argparse     #récupérer l’image en entrée de notre programme
import cv2          #re permettant de gérer les expressions régulières (regex)
import re                       #https://pymotion.com/selectionner-region-interet/
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Chemin de l'image")
args = vars(ap.parse_args())
img = cv2.imread(args["image"])
m = re.search('[^.]*', args["image"])
path = m.group(0)+'_crop.jpg'
ROI = cv2.selectROI(img)

if ROI != (0,0,0,0):
    imgCrop = img[int(ROI[1]):int(ROI[1]+ROI[3]), int(ROI[0]):int(ROI[0]+ROI[2])]
    cv2.imshow("Image", imgCrop)
    cv2.waitKey(0)
    cv2.imwrite(path,imgCrop)

cv2.destroyAllWindows()