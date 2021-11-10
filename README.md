# Loss Re-scaling VQA: Revisiting the Language Prior Problem from a Class-imbalance View
This repository is built upon the [code](https://github.com/hengyuan-hu/bottom-up-attention-vqa). In this paper, we propose a simple loss re-weighting method for tackling the language prior problem in VQA. This repo implements methods on VQA v1 and v2, VQA-CP v1 and v2. I believe everything is simplified in this code and you can easily add your own module! 

The LXMERT version can be found at another repo - https://github.com/guoyang9/LXMERT-VQACP.

Almost all flags can be set by yourself at `utils/config.py`!

This repo also re-implements CSS and LMH, please feel free to take a try. 

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
Note that the loss re-scaling works w/. or w/o. the answer mask pre-training module.

The script for fine-tuning and re-implement our results:
- set `use_mask = True` and `use_miu = False` in `config.py`:
```
python main.py --loss-fn Plain --name test-VQA --gpu 0
```
- fine-tune with the loss re-scaling approach, set both flags with `True`:
```
python main.py --loss-fn Plain --fine-tune --name test-VQA --name-new fine_tune --gpu 0
```

## Model Evaluation
```
python main.py --loss-fn Plain --name test-VQA --eval-only
```
## Citation
```
@article{rescale-vqa,
  title={Loss Re-scaling VQA: Revisiting the Language Prior Problem from a Class-imbalance View},
  author={Guo, Yangyang and Nie, Liqiang and Cheng, Zhiyong and Tian, Qi and Zhang, Min},
  journal={IEEE TIP},
  year={2021}
}
```
```