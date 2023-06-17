import tensorflow as tf
import numpy as np
from tensorflow.compat.v1.keras.backend import set_session
from tensorflow.keras.models import model_from_json

# Configurar el uso de la memoria GPU utilizada para evitar un consumo excesivo de recursos
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.15  # Limitar la cantidad de memoria GPU utilizada por el proceso
session = tf.compat.v1.Session(config=config)
set_session(session)


class FacialExpressionModel(object):

    EMOTIONS_LIST = ['Asco', 'Enojado', 'Feliz', 'Miedo', 'Neutral', 'Sorpresa', 'Triste']

    def __init__(self, model_json_file, model_weights_file):
        # Cargar el modelo desde el archivo JSON
        with open(model_json_file, "r") as json_file:
            loaded_model_json = json_file.read()
            self.loaded_model = model_from_json(loaded_model_json)

        # Cargar los pesos del modelo
        self.loaded_model.load_weights(model_weights_file)

    def predict_emotion(self, img):
        global session
        set_session(session)
        self.preds = self.loaded_model.predict(img)
        return FacialExpressionModel.EMOTIONS_LIST[np.argmax(self.preds)]  # Tomar emoción con la mayor probabilidad