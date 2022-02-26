from PIL import Image
from glitch_this import ImageGlitcher
import cv2
import numpy as np
import os
import random
import statistics


def Cv2ImageToPillowImage(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    return(img)

def PillowImageToCv2Image(img):
    open_cv_image = np.array(img) 
    open_cv_image = open_cv_image[:, :, ::-1].copy() 
    return(open_cv_image)

r = []
g = []
b = []

cascPath=os.path.dirname(cv2.__file__)+"/data/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

glitcher = ImageGlitcher()

camera = cv2.VideoCapture()

camera.open(0, cv2.CAP_DSHOW)

camera.set(cv2.CAP_PROP_FPS, 24)
camWitdh = 1280
camHeight = 720
camera.set(cv2.CAP_PROP_FRAME_WIDTH, camWitdh)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, camHeight)
while(True):
    ret, frame = camera.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    facedetected = False


    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=3,
        minSize=(60, 60),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        dim = (w,h)
        FaceCrop = frame[y:y+h, x:x+w]
        

        downgradeSize = 10

        FaceCrop = cv2.resize(FaceCrop, (downgradeSize,downgradeSize)) ##DOWN GRADE

        for u in range(downgradeSize): 
            for i in range(downgradeSize):
                r.append(FaceCrop[u,i,0])
                g.append(FaceCrop[u,i,1])
                b.append(FaceCrop[u,i,2])
        rMoy = statistics.mean(r)
        gMoy = statistics.mean(g)
        bMoy = statistics.mean(b)
        #print("array : ",rMoy,gMoy,bMoy)

        ### DELETE R G B WHEN ARE TOO LONG 
        if(len(r) >50000):
            r = r[:25000]
            g = g[:25000]
            b = b[:25000]

        #print(len(r))

        if(rMoy > rMoy*0.99 and rMoy<rMoy*1.01 and gMoy > gMoy*0.99 and gMoy<gMoy*1.01 and bMoy>bMoy*0.99 and bMoy<bMoy*1.01):
            facedetected = True
            FaceCrop = cv2.resize(FaceCrop, dim, interpolation = cv2.INTER_NEAREST)

            framePill = Cv2ImageToPillowImage(FaceCrop)

            glitch_img = glitcher.glitch_image(framePill,5,None,0,True,False)

            open_cv_image = PillowImageToCv2Image(glitch_img)

            frame[y:y+h, x:x+w] = open_cv_image
            #cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    if(facedetected == False):
        reducedWidth = int(camWitdh/20)
        reducedHeight = int(camHeight/20)
        frame = cv2.resize(frame, (reducedWidth,reducedHeight) )##DOWN GRADE
        frame = cv2.resize(frame, (camWitdh,camHeight), interpolation = cv2.INTER_NEAREST)
        framePill = Cv2ImageToPillowImage(frame)
        glitch_img = glitcher.glitch_image(framePill,3,None,0,True)
        open_cv_image = PillowImageToCv2Image(glitch_img)
        frame = open_cv_image


    cv2.imshow('Camera',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
camera.release()
cv2.destroyAllWindows()

