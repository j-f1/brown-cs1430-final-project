import os
import tensorflow as tf
from PIL import Image
import numpy as np
import json

from . import hyperparameters as hp
from .models import YourModel

# import hyperparameters as hp
# from models import YourModel

from teeth import are_there_teeth


def predict():
    # Creating model
    model = YourModel()
    model(tf.keras.Input(shape=(hp.img_size, hp.img_size, 3)))
    model.load_weights(
        os.path.join(
            os.path.dirname(__file__),
            "checkpoints/your_model/050322-162304/your.weights.e009-acc0.9450.h5",
        )
    )

    # Assign data path
    data_path = ".." + os.sep + "ai_faces"
    original_images, standardized_data, file_paths = standardize(data_path)

    print("Standardized data shape", standardized_data.shape)

    predictions = model.predict(standardized_data, verbose=1)

    print("Predictions shape", predictions.shape)

    sex = np.argmax(predictions, axis=1)
    print("Sex shape", sex.shape)

    dictionary = {}

    for i in range(len(file_paths)):
        numeric_filter = filter(str.isdigit, file_paths[i])
        file_name = "".join(numeric_filter)
        dictionary[file_name] = {}
        dictionary[file_name]["teeth"] = are_there_teeth(
            (original_images[i] * 255).astype(np.uint8), annotate=False
        )[1]
        if sex[i] == 0:
            dictionary[file_name]["sex"] = "female"
        else:
            dictionary[file_name]["sex"] = "male"

    # the json
    #  where the output must be stored
    out_file = open("../image_data.json", "w")

    json.dump(dictionary, out_file, indent=4)

    out_file.close()


def standardize(data_path):
    """Calculate mean and standard deviation of the images
    for standardization.

        Arguments: none

        Returns: Numpy Array representing standardized images
    """
    # Get list of all images in training directory
    file_list = []
    for root, _, files in os.walk(data_path):
        for name in sorted(files):
            file_list.append(os.path.join(root, name))

    num_files = len(file_list)
    print("NUM FILES", num_files)

    # Allocate space in memory for images
    data_sample = np.zeros((num_files, hp.img_size, hp.img_size, 3))

    # Import images
    for i, file_path in enumerate(file_list):
        if i % 10 == 0:
            print(f"\rReading {i:04}", end="")
        img = Image.open(file_path)
        img = img.resize((hp.img_size, hp.img_size))
        img = np.array(img, dtype=np.float32)
        img /= 255.0

        # Grayscale -> RGB
        if len(img.shape) == 2:
            img = np.stack([img, img, img], axis=-1)

        data_sample[i] = img

    # calculating the mean and std from the training data
    training_path = "." + os.sep + "data" + os.sep + "train" + os.sep
    training_file_list = []
    for root, _, files in os.walk(training_path):
        for name in files:
            training_file_list.append(os.path.join(root, name))

    num_train = len(training_file_list)
    # Allocate space in memory for images
    training_sample = np.zeros((num_train, hp.img_size, hp.img_size, 3))

    for i, file_path in enumerate(training_file_list):
        if i % 10 == 0:
            print(f"\rReading {i:04}", end="")
        img = Image.open(file_path)
        img = img.resize((hp.img_size, hp.img_size))
        img = np.array(img, dtype=np.float32)
        img /= 255.0

        # Grayscale -> RGB
        if len(img.shape) == 2:
            img = np.stack([img, img, img], axis=-1)

        training_sample[i] = img

    mean = np.mean(training_sample, axis=0)
    std = np.std(training_sample, axis=0)

    print(
        "Dataset mean shape: [{0}, {1}, {2}]".format(
            mean.shape[0], mean.shape[1], mean.shape[2]
        )
    )

    print(
        "Dataset mean top left pixel value: [{0:.4f}, {1:.4f}, {2:.4f}]".format(
            mean[0, 0, 0], mean[0, 0, 1], mean[0, 0, 2]
        )
    )

    print(
        "Dataset std shape: [{0}, {1}, {2}]".format(
            std.shape[0], std.shape[1], std.shape[2]
        )
    )

    print(
        "Dataset std top left pixel value: [{0:.4f}, {1:.4f}, {2:.4f}]".format(
            std[0, 0, 0], std[0, 0, 1], std[0, 0, 2]
        )
    )

    original = np.array(data_sample)

    for i in range(num_files):
        img = data_sample[i]
        img = (img - mean) / std
        data_sample[i] = img

    return original, data_sample, file_list


def predict_image(image):
    """Predict sex classificartion for image.

    Arguments: Numpy arrray representing image

    Returns: String representing sex classification
    """

    model = YourModel()
    model(tf.keras.Input(shape=(hp.img_size, hp.img_size, 3)))
    model.load_weights(
        os.path.join(
            os.path.dirname(__file__),
            "checkpoints/your_model/050322-162304/your.weights.e009-acc0.9450.h5",
        )
    )

    img = np.resize(image, (hp.img_size, hp.img_size))
    img = np.array(img, dtype=np.float32)
    img /= 255.0
    print("IMAGE SHAPE", img.shape)

    # Grayscale -> RGB
    if len(img.shape) == 2:
        img = np.stack([img, img, img], axis=-1)

    data_sample = np.zeros((1, hp.img_size, hp.img_size, 3))
    data_sample[0] = img

    # calculating the mean and std from the training data
    training_path = "." + os.sep + "data" + os.sep + "train" + os.sep
    training_file_list = []
    for root, _, files in os.walk(training_path):
        for name in files:
            training_file_list.append(os.path.join(root, name))

    num_train = len(training_file_list)
    # Allocate space in memory for images
    training_sample = np.zeros((num_train, hp.img_size, hp.img_size, 3))

    for i, file_path in enumerate(training_file_list):
        if i % 10 == 0:
            print(f"\rReading {i:04}", end="")
        img = Image.open(file_path)
        img = img.resize((hp.img_size, hp.img_size))
        img = np.array(img, dtype=np.float32)
        img /= 255.0

        # Grayscale -> RGB
        if len(img.shape) == 2:
            img = np.stack([img, img, img], axis=-1)

        training_sample[i] = img

    mean = np.mean(training_sample, axis=0)
    std = np.std(training_sample, axis=0)

    img = data_sample[0]
    img = (img - mean) / std
    data_sample[0] = img

    print("data_sample", img)

    predictions = model.predict(data_sample, verbose=1)
    print(predictions)
    sex = np.argmax(predictions, axis=1)
    print(sex)
    if sex[0] == 0:
        return "female"
    else:
        return "male"


# predict()
# img = Image.open("../ai_faces/099005.png")
# predict_image(img)
