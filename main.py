import numpy as np
import cv2 as cv
from datetime import datetime
import os


def createFolder(nameFolder):
    try:
        os.mkdir(str(nameFolder))
    except:
        pass


def captureBodyAndFace():
    body_cascade = cv.CascadeClassifier('haarcascade_fullbody.xml')

    frontal_face = cv.CascadeClassifier('haarcascade_mcs_upperbody.xml')

    cap = cv.VideoCapture("demo.avi")

    cont = 0
    while(True):
        frameExiste, img = cap.read()
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        bodies = body_cascade.detectMultiScale(gray, 1.3, 5)
        faces = frontal_face.detectMultiScale(gray, 1.3, 5)

        now = datetime.now()
        current_time = now.strftime("%H-%M")

        for (x, y, w, h) in bodies:

            roi_gray = gray[y-20:y+h+20, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            cv.imshow("corpo_cor", roi_color)
            cv.imshow("corpo", roi_gray)

            # salvando
            createFolder(current_time)
            cv.imwrite(str(current_time)+'/corpo-'+str(cont)+'.jpg', roi_color)

            # desenha um retangulo na imagem principal
            cv.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

        for (x, y, w, h) in faces:

            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            cv.imshow("face", roi_color)

            # salvando
            createFolder(current_time)
            cv.imwrite(str(current_time)+'/cara-'+str(cont)+'.jpg', roi_color)

            # desenha um retangulo na imagem principal
            cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cont += 1
        cv.imshow("deteccao", img)

        if cv.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


captureBodyAndFace()
