import numpy as np
import tensorflow as tf
import os
from decouple import config


class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename

    def predict(self):
        # load model
        model_path = config('MODEL_PATH', default="/Users/tisha/Downloads/A_EXAM/ME_4/CE6/M_M/Kidney-Disease-Classification/artifacts/training")
        model = tf.keras.models.load_model(os.path.join(model_path, "model.h5"))
        imagename = self.filename
        test_image = tf.keras.utils.load_img(imagename, target_size=(224, 224))
        test_image = tf.keras.utils.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        result = np.argmax(model.predict(test_image), axis=1)
        print(result)

        # if result[0] == 1:
        #     prediction = 'Tumor'
        #     return [{"image": prediction}]
        # else:
        #     prediction = 'Normal'
        #     return [{"image": prediction}]
        
        
        if result[0] == 1:
            prediction_text = 'Tumor'
            return [{
                "PREDICTION": "Our AI Predicts Tumor."
            }]
        else:
            prediction_text = 'Normal'
            return [{
                "PREDICTION": "No Abnormalities Detected."
            }]


