# 청계산셰르파의 KLUE RE

<details>
  <summary> 업데이트로그 </summary>

    dataset 추가
        - dataset 경로가 제각각인데, 생각보다 데이터셋 용량이 작길래 그냥 git에 전부 올려버렸습니다. 경로설정하는것보단 편할 것 같아서요 ㅎㅎ.
        - 10%, 15%, 20%로 stratified 하게 나눈 train/valid dataset과 약 1500개/130개 정도의 small train/valid 데이터셋, 그리고 하겸님이 작업해주신 특수문자가 제거된 preprocess train/test 데이터셋을 data 폴더에 포함시켰습니다.
        - 관련해서 각 파일별로 지정되어있던 default dataset 경로를 data/~로 일괄적으로 수정하였습니다.
        - 또한 data 폴더를 만들면서 pkl 이녀석들도 전부 data로 옮겨버렸습니다.

    argument 추가
        - eval_strategy 라는 argument를 추가했습니다. steps와 epoch 별로 evaluation 전략을 다르게 하기 위해서입니다. 이에 따라 eval_strategy 값에 따라서 training argument를 다르게 할 수 있도록 분기를 나누었습니다.

    init.sh 추가
        - 데이터셋 전처리를 한 번에 하거나 커밋 메세지 템플릿 일괄적용하거나 다른 공통사항들을 추가할 때 편하게 하기 위해 bash script를 만들긴 했는데 추가적으로 만들려고 하다보니 생각보다 쓸모가 없어진 것 같아서 하다 말았습니다. 추후에 필요해지면 그 때 이어서 추가적으로 작업해보겠습니다.

    inference_kfold.py
        - kfold 작업 후  inference 함수가 없어서 민수님 버전에 대응되도록 inference_kfold.py 를 만들었습니다.
        - 이 때 train_kfold에서 저장한 모델 경로가 inference_kfold 에서 불러오는 모델 경로와 같아야합니다. 주의요망!

    requirements 업데이트 및 git message template 추가, gitignore 업데이트
        - 자잘한 녀석들 fix해주었습니다.

    그 외
        - 빠른 테스트를 위해 default를 small 데이터셋으로 해놨습니다. 주의요망!!

</details>

## Init

```shell
bash init.sh
```

## Train & Inference

### default
1. edit arguments in train.py
2. `python train.py`
3. edit arguments in inference.py
4. `python inference.py`

### with kfold
1. edit arguments in train_kfold.py
2. `python train_kfold.py`
3. edit arguments in inference_kfold.py
4. `python inference_kfold.py`