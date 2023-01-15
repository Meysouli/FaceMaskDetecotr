# Created by Oumaima Souli.

import numpy as np
from keras.models import load_model
import keras.utils as image
import cv2
import datetime
import paho.mqtt.client as paho
from paho import mqtt
from train import train

# setting HiveMQ callback for the connection event
def on_connect(client, userdata, flags, rc, properties=None):
    print("Connection received with code %s.." % rc)

# setting HiveMQ callback for the publish event
def on_publish(client, userdata, mid, properties=None):
    print("mid: " + str(mid))

# connect to hiveMQ network
client = paho.Client(client_id="", userdata=None, protocol=paho.MQTTv5)

# setting callbacks
client.on_connect = on_connect
client.on_publish = on_publish

# enable TLS for secure connection
client.tls_set(tls_version=mqtt.client.ssl.PROTOCOL_TLS)
# set username and password
client.username_pw_set("Face_Mask_Detector", "Qwerty@123")
# connect to HiveMQ Cloud on port 8883 (default for MQTT)
client.connect("5484ff1d6c0547c482a4b1b3d5610698.s2.eu.hivemq.cloud", 8883)

# start the loop
client.loop_start()


# Train the model based on provided data
# uncomment the next line in case of a first run in order to generate the trained model
# train()

# Load trained model
model = load_model('model.h5')

# Start the live capturing
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
while cap.isOpened():
    _, img = cap.read()
    face = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=4)
    for (x, y, w, h) in face:
        face_img = img[y:y + h, x:x + w]
        cv2.imwrite('temp.jpg', face_img)
        test_image = image.load_img('temp.jpg', target_size=(150, 150, 3))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        pred = model.predict(test_image)[0][0]
        if pred == 1:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)
            cv2.putText(img, 'NO MASK', ((x + w) // 2, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            # publish false after scanning a face without a mask
            client.publish("face/mask-detector", payload="false", qos=1)
            print("published with false")
        else:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(img, 'MASK', ((x + w) // 2, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            # publish true after scanning a face with a mask
            client.publish("face/mask-detector", payload="true", qos=1)
            print("published with true")

        current_datetime = str(datetime.datetime.now())
        cv2.putText(img, current_datetime, (400, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow('img', img)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
