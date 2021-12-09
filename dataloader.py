import os
import json
import random
import numpy as np
from os.path import join as opj
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path

# For 475, 543 folders in train dataset
def remove_keypoints(folder_num, points):
    lst = []
    for x,y,z in points:
        cond1 = x<250 and y>800
        cond2 = x>1400 and y<400
        if not (cond1 or cond2):
           lst.append([x,y,z]) 
    # print('Finished removing {} wrong keypoints....'.format(folder_num))
    return lst

class Train_Dataset(Dataset):
    def __init__(self, df, transform=None, df_flip_info=None, flipaug_ratio=0, label_encoder=None, margin=50, random_margin=True):
        self.id = df['train_path'].values
        self.target = df['answer'].values
        self.transform = transform
        self.margin = margin
        self.random_margin = random_margin

        # Flip Augmentation (Change target class)
        if df_flip_info is not None:
            self.use_flip = True
            print('Use Flip Augmentation')
            left = label_encoder.transform(df_flip_info['left'])
            right = label_encoder.transform(df_flip_info['right'])
            left_to_right = dict(zip(left, right))
            right_to_left = dict(zip(right, left))
            
            self.flip_info = left_to_right.copy()
            self.flip_info.update(right_to_left)        
            self.flip_possible_class = list(set(np.concatenate([left, right])))
        self.flipaug_ratio = flipaug_ratio

        print(f'Dataset size:{len(self.id)}')

    def __getitem__(self, idx):
        image = np.array(Image.open(self.id[idx]).convert('RGB'))
        target = self.target[idx]

        # Load Json File
        try:
            image_num = int(Path(self.id[idx]).stem)
            dir = os.path.dirname(self.id[idx])
            folder_num = os.path.basename(dir)
            json_path = opj(dir, folder_num+'.json')
            js = json.load(open(json_path))
            keypoints = js['annotations'][image_num]['data']  # 해당 이미지에 해당하는 Keypoints
        except:  # Augmentation으로 직접 새로 만든 Folder는 Json이 없으므로
            image = self.transform(Image.fromarray(image))
            return image, np.array(target)

        if folder_num in ['475', '543']:
            keypoints = remove_keypoints(folder_num, keypoints)

        # Image Crop using keypoints
        max_point = np.max(np.array(keypoints), axis=0).astype(int) + self.margin
        min_point = np.min(np.array(keypoints), axis=0).astype(int) - self.margin
        max_point = max_point[:-1] # remove Z order
        min_point = min_point[:-1] # remove Z order

        max_x, max_y = max_point
        min_x, min_y = min_point
        max_y += 100  # 손목까지
        
        if self.random_margin:  # Train Phase
            # 한 폴더 비슷한 이미지들을 조금씩 다르게 잘리도록 (Train만)
            if random.random() < 0.5:
                max_x += self.margin
            if random.random() < 0.5:
                max_y += self.margin
            if random.random() < 0.5:
                min_x -= self.margin
            if random.random() < 0.5:
                min_y -= self.margin
        else:
            max_x += self.margin
            max_y += self.margin
            min_x -= self.margin
            min_y -= self.margin

        # 데이터 포인트의 크기가 원 이미지를 넘어서는 경우를 방지
        max_x = max_x if max_x < 1920 else 1920
        max_y = max_y if max_y < 1080 else 1080
        min_x = min_x if min_x > 0 else 0
        min_y = min_y if min_y > 0 else 0
        
        image = image[min_y:max_y, min_x:max_x]

        # FlipAug
        if (random.random() < self.flipaug_ratio) and (target in self.flip_possible_class):
            image = np.flip(image, axis=1)  # (H, W, C)에서 width 축 flip
            target = self.flip_info[target]

        image = self.transform(Image.fromarray(image))
        return image, np.array(target)

    def __len__(self):
        return len(self.id)

def get_loader(df, batch_size, shuffle, num_workers, transform, df_flip_info=None, 
                flipaug_ratio=0, label_encoder=None, margin=50, random_margin=True):
    dataset = Train_Dataset(df, transform, df_flip_info=df_flip_info, flipaug_ratio=flipaug_ratio, 
                            label_encoder=label_encoder, margin=margin, random_margin=random_margin)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True,
                                drop_last=False)
    return data_loader

def get_train_augmentation(img_size, ver):
    if ver==1:
        transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                ])


    if ver==2:
        transform = transforms.Compose([
                transforms.RandomAffine(20),
                transforms.RandomPerspective(),
                transforms.ToTensor(),
	            transforms.Resize((img_size, img_size)),
    	        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
            ])

    return transform

