## Setup

Make sure to have Python 3.9 and Node.js 18.0.0 installed

```shellsession
$ python3.9 -m venv venv
$ . venv/bin/activate
$ pip install -r requirements.txt
$ cd cnn_code
$ pip install -r requirements.txt
$ wget https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/gender.caffemodel
$ cd ..
```

Then, in one terminal:

```shellsession
$ cd server
$ flask run
```

…and in another:

```shellsession
$ cd frontend
$ npm install
$ npm start
```

## Web UI

1. In one terminal, run `flask run` in the `server/` folder
2. In another terminal, run `npm install` and then `npm start` in the `frontend/` folder

## How to run tutorial at the moment
pip install -r requirements.txt
Download the gender.caffemodel in the link below and place in cnn_code directory

## Resources

- https://pysource.com/2019/05/28/face-swapping-explained-in-8-steps-opencv-with-python/
- https://pypi.org/project/dlib/
- https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html

## Datasets

- https://www.kaggle.com/datasets/dadajonjurakuziev/movieposter
- https://www.kaggle.com/datasets/xuejunz/film-posters-information
- https://www.kaggle.com/datasets/spiyer/old-film-restoration-dataset
- https://www.kaggle.com/datasets/cashutosh/gender-classification-dataset/code (could be used for training a model for sex classification)
- https://www.kaggle.com/gpiosenka/gender-classification-from-an-image (smaller sex data set, in repo currently)
- https://drive.google.com/drive/folders/1-0YhtXe_oE2ei0R471X33a_NJyY5dVge


## Meetings w/ Ta

### April 15

- look into pretrained facial feature models
  - https://pypi.org/project/dlib/
- guessing sex
  - might need to train a secondary model
  - start by not worrying about this, mismatches are ok
  - instead get it working good first
  - later add a second level of detection
  - also don’t worry about glasses too much
- focus on blending well with skin instead (/ “does it look like it’s the actual person’s face”)
- worth a try with skin tone
- TA will look into preexisting models
- facial expression / emotion may impact results
  - consider finetuning pre-existing models
    - take an existing model and then add a few layers onto the end of it, highly specialized to their dataset
    - train the new layers using a small dataset
    - start w/ pytorch tutorials (which go step by step through this process)
    - https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
- do not worry if it doesn’t work, we’re supposed to explore / do something cool
  - more important: did we put time/effort in?
  - did we try different things?
  - writeup is very important
