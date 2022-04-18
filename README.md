[jed] this week: build a fancy web ui

## Resources

- https://pysource.com/2019/05/28/face-swapping-explained-in-8-steps-opencv-with-python/
- https://pypi.org/project/dlib/
- https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html

## Datasets

- https://www.kaggle.com/datasets/dadajonjurakuziev/movieposter
- https://www.kaggle.com/datasets/xuejunz/film-posters-information
- https://www.kaggle.com/datasets/spiyer/old-film-restoration-dataset


## Meetings w/ Ta

### April 15

- look into pretrained facial feature models
  - https://pypi.org/project/dlib/
- guessing gender
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
