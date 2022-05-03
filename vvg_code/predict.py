import os
import tensorflow as tf

import hyperparameters as hp
from models import YourModel
from preprocess import Datasets

def main():
    #creating model
    model = YourModel()
    model(tf.keras.Input(shape=(hp.img_size, hp.img_size, 3)))
    model.load_weights(
    "checkpoints/your_model/042822-220607/your.weights.e012-acc0.9350.h5")
    
    #creating data processor
    data_path = '..'+os.sep+'ai_faces'
    data_processor = Datasets(data_path, "1")
    test_data = data_processor.get_data(data_path, False, False, False)
    print("Test data length", test_data.__len__())
    predictions = model.predict(test_data)
    
main()