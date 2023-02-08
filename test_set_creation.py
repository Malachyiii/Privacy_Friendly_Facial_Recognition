import os
import cv2 as cv
import time

cap = cv.VideoCapture(0)

while(cap.isOpened()==False):
    print("Waiting for connection...")
    time.sleep(5)

print("Video Connected...")

face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
print("Face Detector Created")

if not os.path.exists("./TrainingSet/"):
    os.makedirs("./TrainingSet")

set_creation = True

while(set_creation):
    
    new_person = input("Add another person [y/n]?")
    
    if new_person.lower() == 'n':
        set_creation = False
        pass
    
    elif new_person.lower() == 'y':
        person = input("Who is this? ").lower()
        count = int(input(f"How Many Images would you like to take of {person}? "))
        images = 0
        
        while images < count:
            ret, frame = cap.read()
            # gray here is the gray frame you will be getting from a camera
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x,y,w,h) in faces:
                print("Face seen")
                face = frame[y:y+h, x:x+w]
                face = cv.resize(face, (256,256), interpolation = cv.INTER_LINEAR)
                cv.imwrite(f"./TrainingSet/{person}_{images}.png", face)
                images += 1
            
            #cv.imshow('frame',gray)
        
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

cap.release()
cv.destroyAllWindows()
