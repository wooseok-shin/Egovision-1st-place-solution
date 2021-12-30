# Egovision-1st-place-solution
This repository is the 1st place solution for [DACON Ego-Vision Hand Gesture Recognition AI Contest](https://dacon.io/competitions/official/235805/overview/description).

## Overview
본 대회를 참가하기 이전에도 데이콘을 포함하여 다양한 컴페티션을 참여하면서 Model이나 Training technique을 적용하는 위주로 대회를 참여했었습니다.  
이번 대회를 계기로 다시 한 번 데이터를 직접 보면서 분석하고 살펴보는 것이 중요하다는 것을 느낄 수 있는 대회였습니다.

- Image(with keypoint)를 사용한 Classification 모델을 Base로 많은 방법들을 시도해보았고, 확실한 효과를 봤던 기법들을 순서대로 정리하자면 아래와 같습니다.
    - Rule-based Approach : Public-0.00670, Private score-0.00578 (아래에서 자세히 분석하겠지만 큰 효과를 봤습니다.)
    - Flip Augmentation
    - Random Margin Crop Image
    - Random Affine & Random Perspective Augmentation
    - OneCycleLR Scheduler
    - WarmUpLR Scheduler

<br>

- 최종적으로 제출한 2가지는 5Fold Ensemble 결과 하나와 Validation이 가장 잘나온 Single Fold 모델 하나(추론 시간까지 고려했었습니다)를 제출했습니다.
- Private 결과가 나왔을 때는 신기하게도 Single Fold 모델 하나가 선택되었습니다. 본 대회가 테스트셋이 217개로 매우 적다보니 모델 결과에도 운이 꽤 작용하는 것 같습니다.
- 다만 안정적인 재현을 위해 해당 코드는 5Fold Ensemble로 작성하였고, Best Single Fold(0~4폴드 중 4폴드)모델은 Github에서 Pretrained Weight을 불러오도록 하였습니다.
- 최종 결과: 5Fold Ensemble - Public:0.00627 / Single Best - Public:0.00670, Private:0.00578    (대회가 끝난 후 5Fold의 Private 결과는 알 수 없는 점이 조금 아쉬웠습니다.)


## Requirements
- Ubuntu 18.04, Cuda 11.1
- opencv-python  
- numpy  
- pandas
- timm
- torch==1.8.0 torchvision 0.9.0 with cuda 11.1
- natsort
- scikit-learn==1.0.0
- pillow
- torch_optimizer
- tqdm
- ptflops
- easydict
- matplotlib

```python
pip install -r requirements.txt
```

## Make dataset
```python
python make_data.py
```

## Train
```python
#1 Single Model Train
python main.py --img_size=288 --exp_num=0

#2 Multiple training using shell
sh multi_train.sh
```

## Make a prediction with post-processing
```python
# Pretrained weight download from github
os.makedirs('./results/', exist_ok=True)
!wget -i https://raw.githubusercontent.com/wooseok-shin/Egovision-1st-place-solution/main/load_pretrained.txt -P results
```

```python
python test_post_processing.py
```


