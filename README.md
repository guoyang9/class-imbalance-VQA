# Loss-rescaling VQA: Revisiting Language Priors via A Class-imbalance View
This repository is built upon the [code](https://github.com/Cyanogenoid/vqa-counting.git) provided by @Yan Zhang. Futher introduction will be given shortly.

Almost all flags can be set by yourself at `utils/config.py`!

## Repo-download
```
git clone --recursive https://github.com/guoyang9/class-imbalance-VQA.git
```
## Prerequisites
    * python==3.7.7
    * nltk==3.4
    * bcolz==1.2.1
    * tqdm==4.31.1
    * numpy==1.18.4
    * pytorch==1.4.0
    * tensorboardX==2.1
    * torchvision==0.6.0
## Dataset
First of all, make all the data in the right position according to the `utils/config.py`!

* Please download the VQA-CP datasets in the original paper.
* The image features can be found at the UpDn repo.
* The pre-trained Glove features can be accessed via [GLOVE](https://nlp.stanford.edu/projects/glove/).


## Pre-processing

1. process questions and dump dictionary:
    ```
    python tools/create_dictionary.py
    ```

2. process answers and question types:

    ```
    python tools/compute_softscore.py
    ```
3. convert image features to h5:
    ```
    python tools/detection_features_converter.py 
    ```
## Model Training
```
python main.py --loss-fn Plain --name test-VQA --gpu 0
```

## Model Evaluation 
```
python main.py --loss-fn Plain --name test-VQA --eval-only
```
## Citation
