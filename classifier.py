import numpy as np
import cv2 as cv
import paho.mqtt.client as mqtt
import time

LOCAL_MQTT_HOST="mosquitto-service"
LOCAL_MQTT_PORT=1883
LOCAL_MQTT_TOPIC="detected"

cap = cv.VideoCapture(0)

while(cap.isOpened()==False):
    print("Waiting for connection...")
    time.sleep(5)

print("Video Connected...")

face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
print("Classifier Created")

is_connected = 0

def on_connect_local(client, userdata, flags, rc):
        print("connected to local broker with rc: " + str(rc))
        client.subscribe(LOCAL_MQTT_TOPIC)
        if rc ==0:
            global is_connected
            is_connected = 1


local_mqttclient = mqtt.Client()
local_mqttclient.on_connect = on_connect_local
local_mqttclient.connect(LOCAL_MQTT_HOST, LOCAL_MQTT_PORT, 60)
local_mqttclient.loop_start()
print("is connected: " + str(is_connected))


while(True):
    ret, frame = cap.read()
    # gray here is the gray frame you will be getting from a camera
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        print("Face seen")
        # your logic goes here; for instance
    	# cut out face from the frame..
        #cv.rectangle(gray,(x,y),(x+w,y+h),(255,0,0),2)
        #cv.imshow('frame',gray)
        face = frame[y:y+h, x:x+w]
        rc,png = cv.imencode('.png', face)
        msg = png.tobytes()
        print(str(msg))
        local_mqttclient.publish(LOCAL_MQTT_TOPIC, msg)
    
    #cv.imshow('frame',gray)
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()

local_mqttclient.loop_stop()
