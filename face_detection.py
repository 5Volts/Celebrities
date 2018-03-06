import cv2
import sys
import time
import pickle
import glob

def detect(image):
    li = []
    cascPath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascPath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags = cv2.CASCADE_SCALE_IMAGE
    )
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (200, 200, 0), 20)
        crop_img = image[y+15:y + h-15, x+15:x + w-15]
        cv2.imshow("Tit",image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        li.append(image)
        li.append(crop_img)
    try:
        return li
    except UnboundLocalError:
        return "No Faces found"

def check(image):
    try:
        data=detect(image)
        if data == "No Faces found":
            print("No faces found")
        else:
            return data[1]
    except cv2.error:
        print("Error")



