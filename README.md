# Boostcamp KLUE Relation Extraction Competition
Code for 5th place in Boostcamp Klue RE comepetition. 3rd place by official best model.

For detailed documentation visit [here](https://docs.google.com/document/d/1cTlZ_MsTGlkTU0ZAuWmexvIkz2vbvZrViruU64MQ3fs/edit?usp=sharing)

## Hardware
- NVIDIA TELSA V100
- Ubuntu 18.04

## Reproduction of the Best Model
You can reproduce the submission by retraining through the following steps:
- [Installation](#Installation)
- [Preprocessing](#Preprocessing)
- [Training](#Training)
- [Inference](#Inference)

## Background Info
### Installation

```
conda create -n klue_re python=3.6
conda activate klue_re
pip install -r requirements.txt
```


### Data Source
In this competition, we used KLUE dataset for relation extraction, which can be downloaded from the link below:
https://aistages-prod-server-public.s3.amazonaws.com/app/Competitions/000075/data/dataset.tar.gz

### EDA
EDA can be seen from eda.ipynb file under data_eda_preprocessing folder

### Preprocessing
Multiple augmentation techniques were applied but only few has proven valid. Targeting and augmenting the data that the 5-fold model was at least wrong once was effective. All other experiments were futile, although the experiments included techniques commonly accepted effective. Such techniques include back-translation, changing word seuqence, adding punctuation, replacing words with synonyms, and adding new words (Karimi et al.; wei et al.).

To get the final data we used, run
`cleanser.ipnyb` to get a cleansed data `pristine.csv`, and use the file to run
`target_augment.ipynb` to get a selectively augmented data, which is `cleansed_target_augmented.csv`.

Using `stratified_sentence_split.ipynb` will produce a k-fold stratified data that does not have same sentences between train and validation set, but this technique has proven to not have much improvement.

## HOW TO USE

### Training
#### default
`python train.py`

#### w/ stratified k fold
`python train.py --mode skf`

#### if you want to repeat current best score
`python train.py --mode skf --hp_config 'hp_config/roberta_large_focal_loss.json' --model_name 'klue/roberta-large'`

#### if you want to use augmented dataset,
`python train.py --aug_data_dir 'data/eda_repeat_1.csv'`


### Inference
#### default
`python inference.py`
#### w/ stratified k fold
`python inference.py --mode skf`


## Reference
Karimi, Akbar, Leonardo Rossi, and Andrea Prati. "AEDA: An Easier Data Augmentation Technique for Text Classification." arXiv preprint arXiv:2108.13230 (2021). <br>
Wei, Jason, and Kai Zou. "Eda: Easy data augmentation techniques for boosting performance on text classification tasks." arXiv preprint arXiv:1901.11196 (2019).
