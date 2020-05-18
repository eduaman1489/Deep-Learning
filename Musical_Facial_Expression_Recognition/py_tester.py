import numpy as np
import cv2
from keras.models import model_from_json
from keras.preprocessing import image
import os
import random
import vlc   
import time



face_cascade=cv2.CascadeClassifier('/Users/heisenberg/deep_learning_code/Facial_expression_recognition/haarcascade_frontalface_alt2.xml')

#load saved model
model = model_from_json(open('/Users/heisenberg/deep_learning_code/Facial_expression_recognition/model_weights/fer_colab.json', 'r').read())

#load weights
model.load_weights('/Users/heisenberg/deep_learning_code/Facial_expression_recognition/model_weights/fer_colab.h5')

    
cap=cv2.VideoCapture(0)

while True:
    ret,test_img=cap.read()                  # captures frame and returns boolean value and captured image
    if not ret:
        continue
    gray_img= cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

    faces_detected = face_cascade.detectMultiScale(gray_img, 1.32, 5)

    play_song=[]
    direc_path=[]
    
    for (x,y,w,h) in faces_detected:
        cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=7)
        roi_gray=gray_img[y:y+w,x:x+h]       #cropping region of interest i.e. face area from  image
        roi_gray=cv2.resize(roi_gray,(48,48))
        img_pixels = image.img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis = 0)
        img_pixels /= 255

        predictions = model.predict(img_pixels)

        #find max indexed array
        max_index = np.argmax(predictions[0])

        emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
        predicted_emotion = emotions[max_index]
        print(predicted_emotion)
        cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        
        
        dir_path=os.path.join(os.getcwd(),'Music',predicted_emotion)
        direc_path.append(dir_path)
        song=[f for f in os.listdir(dir_path) if f.endswith('.mp3')]
        song=random.choice(song)
        song_path=(direc_path[0]+'/'+song)
        
        
        p = vlc.MediaPlayer(song_path)
        p.play()
        time.sleep(10)
        p.stop()

        
    resized_img = cv2.resize(test_img, (1000, 700))
    cv2.imshow('Facial emotion analysis ',resized_img)
    
    if cv2.waitKey(20) == ord('q'):           #wait until 'q' key is pressed
        break

cap.release()
cv2.destroyAllWindows