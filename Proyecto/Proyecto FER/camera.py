import cv2  # Usar openCV
import numpy as np
from model import FacialExpressionModel

# Cargar el clasificador de cascada Haar para la detección de rostros
facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Cargar el modelo de expresión facial previamente entrenado
model = FacialExpressionModel("model.json", "model_weights.h5")

# Definir la fuente para el texto que se mostrará en las imágenes
font = cv2.FONT_HERSHEY_SIMPLEX

# Dirección del video (cambiar)
video_path = r"C:\Users\User\Documents\0. Semestres\2023-1\Inteligencia artificial\Proyecto\Proyecto FER\videos\Duque.mp4"  

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(video_path)

    def __del__(self):
        self.video.release()
    
    # Muestra cada fotograma señalando la cara y la predicción
    def get_frame(self):
        _, fr = self.video.read()
        gray_fr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        faces = facec.detectMultiScale(gray_fr, 1.3, 5)

        for (x, y, w, h) in faces:
            fc = gray_fr[y:y+h, x:x+w]

            roi = cv2.resize(fc, (48, 48))
            pred = model.predict_emotion(roi[np.newaxis, :, :, np.newaxis])

            cv2.putText(fr, pred, (x, y), font, 1, (255, 255, 0), 2)
            cv2.rectangle(fr,(x,y),(x+w,y+h),(255,0,0),2)

        _, jpeg = cv2.imencode('.jpg', fr)
        return jpeg.tobytes()


