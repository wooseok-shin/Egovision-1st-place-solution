#%%
import os
import cv2
from glob import glob
import json
import random
import numpy as np
import pandas as pd
from natsort import natsorted
from tqdm import tqdm
from os.path import join as opj
from crop_image import crop_image
# dataloader에서 사용할 dataframe 만들기
###################################
train_path = '../data/train'
train_folders = natsorted(glob(train_path + '/*'))

answers = []
for train_folder in train_folders:
    json_path = glob(train_folder + '/*.json')[0]
    js = json.load(open(json_path))
    cat = js.get('action')[0]
    cat_name = js.get('action')[1]
    
    images_list = glob(train_folder + '/*.png')
    for image_name in images_list:
        answers.append([image_name, cat, cat_name])

answers = pd.DataFrame(answers, columns = ['train_path','answer', 'answer_name'])
answers.to_csv('../data/df_train.csv', index=False)
#%%
##################################
# 클래스가 1개씩 있는 폴더 Augmentation해서 폴더 생성하기
seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)

data_path = '../data'
df_train = pd.read_csv(opj(data_path, 'df_train.csv'))
df_info = pd.read_csv(opj(data_path, 'hand_gesture_pose.csv'))
df_train = df_train.merge(df_info[['pose_id', 'gesture_type', 'hand_type']],
                        how='left', left_on='answer', right_on='pose_id')

save_folder = 'train' 
for i in range(649, 649+5):
    if not os.path.exists(opj(data_path, save_folder, str(i))):
        os.makedirs(opj(data_path, save_folder, str(i)))

# flip aug가능한 label : 131, 47 (one sample)
oslabel_fliplabel = [(131,156), (47, 22)] # one sample label, flip label
folders = ['649', '650'] # target folder
for label, folder in tqdm(zip(oslabel_fliplabel, folders)):
    idx = 0
    os_label, f_label  = label[0], label[1]
    one_sample = df_train[df_train['answer'] == os_label].reset_index(drop=True)
    temp = df_train[df_train['answer'] == f_label].reset_index(drop=True)
    train_folders = natsorted(temp['train_path'].apply(lambda x : x[:-6]).unique())
    for train_folder in (train_folders):
        json_path = glob(train_folder + '/*.json')[0]
        js = json.load(open(json_path))
        keypoints = js['annotations']
        images_list = natsorted(glob(train_folder + '/*.png'))
        for _, (point, image_name) in enumerate(zip(keypoints, images_list)):
            croped_image = crop_image(image_name, point, margin=50)
            flip_img = cv2.flip(croped_image, 1)
            save_path = opj(data_path, save_folder, folder, f'{idx}.png')
            idx += 1
            cv2.imwrite(save_path, flip_img)
            df_train.loc[len(df_train)] = [save_path] + one_sample.iloc[0][1:].values.tolist()

# flip aug불가능한 label -> affine 통해 img agu
def rotation(img, angle):
    angle = int(random.uniform(-angle, angle))
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((int(w/2), int(h/2)), angle, 1)
    img = cv2.warpAffine(img, M, (w, h)) 
    return img

oslabel = [92, 188, 145]
folder = ['651', '652', '653']
for label, folder in tqdm(zip(oslabel, folder)):
    idx = 0
    one_sample = df_train[df_train['answer'] == label].reset_index(drop=True)
    train_folders = natsorted(temp['train_path'].apply(lambda x : x[:-6]).unique())
    for train_folder in (train_folders):
        json_path = glob(train_folder + '/*.json')[0]
        js = json.load(open(json_path))
        keypoints = js['annotations']
        images_list = natsorted(glob(train_folder + '/*.png'))
        for _, (point, image_name) in enumerate(zip(keypoints, images_list)):
            croped_image = crop_image(image_name, point, margin=50)
            aug_img = rotation(croped_image, 30)
            save_path = opj(data_path, save_folder, folder, f'{idx}.png')
            idx += 1
            cv2.imwrite(save_path, aug_img)
            df_train.loc[len(df_train)] = [save_path] + one_sample.iloc[0][1:].values.tolist()

df_train.to_csv('../data/df_train_add.csv', index=False)