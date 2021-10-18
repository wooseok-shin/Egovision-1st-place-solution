# Egovision-1st-place-solution
This repository is the 1st place solution for [DACON Ego-Vision Hand Gesture Recognition AI Contest](https://dacon.io/competitions/official/235805/overview/description).



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
python main.py --img_size=256 --exp_num=0

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


