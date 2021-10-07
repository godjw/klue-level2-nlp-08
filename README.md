# HOW TO USE

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
