#%%
import numpy as np
from natsort import natsorted
from glob import glob
from tqdm import tqdm
import json

def check_stats(find_list, ver):
    train_path = '../data/train'
    train_folders = natsorted(glob(train_path + '/*'))
    stat_list = []
    for _, train_folder in tqdm(enumerate(train_folders)):
        try:
            json_path = glob(train_folder + '/*.json')[0]
            js = json.load(open(json_path))
            cat = js.get('action')[0]
            keypoints = js['annotations']
            keypoints = np.array([point['data'] for point in keypoints])  # (N-이미지개수, 21 or 42(keypoints), 3(x,y,z 좌표))
        except:
            pass
            # print(train_folder)
        if cat in find_list:
            # Case1 : 숫자1과 검지흔들기 구분
            # 검지는 이미지내 keypoint들 중 가장 낮은 y값(이미지 상 가장 높은 위치)을 갖는 point임.
            if ver ==1 : 
                keypoints = keypoints[:, :, :2] 
                x_I_finger = [point_per_img[:,0][point_per_img[:,1].argmin()] for point_per_img in keypoints] 
                stat_list.append(np.max(x_I_finger) - np.min(x_I_finger))
            

            # Case2 : 주먹쥐기와 주먹 내밀기(경고)
            elif ver == 2:
                keypoints = keypoints[:, :, 0]  
                x_values = [point_per_img[point_per_img.argmax()] for point_per_img in keypoints] # 가장 오른쪽 point
                stat_list.append(np.max(x_values) - np.min(x_values))
            
    return stat_list
#%%
############ ver1 ###############
find_list0 = [0, 10, 100, 110] # ['숫자 1', '숫자1']  my hand, your hand 좌우
find_list1= [42, 67, 142, 167] # ['부정(검지 흔들기)'] my hand, your hand 좌우

############ ver2 ###############
find_list2 = [146] # ['주먹쥐기']  Your hand 우
find_list3 = [163] # ['경고(주먹 내밀기)'] Your hand 우

find_list4 = [171] # ['주먹쥐기']  Your hand Both
find_list5 = [191] # ['경고(주먹 내밀기)'] Your hand Both
#%%
# 숫자1 & 검지 흔들기
li0 = check_stats(find_list0,1)  
li1 = check_stats(find_list1,1)   
threshold_ver1 = max(li0) + 5 # 31.8505859375

# 주먹쥐기 vs 주먹 내밀기 Right
li2 = check_stats(find_list2,2)   
li3 = check_stats(find_list3,2)  
threshold_ver2 = max(li2) + 5 # 29.80083465576172

# 주먹쥐기 vs 주먹 내밀기 Both
li4 = check_stats(find_list4,2)   
li4 = li4[1:]
li5 = check_stats(find_list5,2)  
threshold_ver2_both = max(li4) + 5 # 34.33704376220703
#%%
def Refiner(keypoints, ver):
    keypoints = np.array([point['data'] for point in keypoints]) 
    # 숫자1과 검지흔들기 구분
    if ver == 1:  
        keypoints = keypoints[:, :, :2]  
        x_I_finger = [point_per_img[:,0][point_per_img[:,1].argmin()] for point_per_img in keypoints]
        query_value = np.max(x_I_finger) - np.min(x_I_finger)
    
    # 주먹쥐기와 주먹 내밀기(경고) 
    elif ver == 2:
        keypoints = keypoints[:, :, 0]  
        x_values = [point_per_img[point_per_img.argmax()] for point_per_img in keypoints]
        query_value = np.max(x_values) - np.min(x_values)
    
    return query_value