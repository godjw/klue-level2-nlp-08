# HOW TO USE

## Data Source
In this competition, we used KLUE dataset for relation extraction, which can be downloaded from the link below:
https://aistages-prod-server-public.s3.amazonaws.com/app/Competitions/000075/data/dataset.tar.gz

## EDA
EDA can be seen from eda.ipynb file under data_eda_preprocessing folder

## Preprocessing
Multiple 



## Training
### default
`python train.py`
### w/ stratified k fold
`python train.py --mode skf`

### if you want to repeat current best score
`python train.py --mode skf --hp_config 'hp_config/roberta_large_focal_loss.json' --model_name 'klue/roberta-large'`

### if you want to use augmented dataset,
`python train.py --aug_data_dir 'data/eda_repeat_1.csv'`


## Inference
### default
`python inference.py`
### w/ stratified k fold
`python inference.py --mode skf`
