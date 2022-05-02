import os
import sys
import argparse
import re
from datetime import datetime
import tensorflow as tf

import hyperparameters as hp
from models import YourModel
from preprocess import Datasets
from skimage.transform import resize

from skimage.io import imread
from lime import lime_image
from skimage.segmentation import mark_boundaries
from matplotlib import pyplot as plt
import numpy as np

def main():
    print("WORKING")
    model = YourModel()
    model(tf.keras.Input(shape=(hp.img_size, hp.img_size, 3)))
    model.load_weights(
    "checkpoints/your_model/042822-220607/your.weights.e012-acc0.9350.h5")
    
main()