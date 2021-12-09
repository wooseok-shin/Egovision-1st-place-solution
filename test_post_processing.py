#%%
import os
import pandas as pd
import torch
import torchvision.transforms as transforms
from os.path import join as opj
from tqdm import tqdm
from glob import glob
from easydict import EasyDict
from natsort import natsorted
from dataloader import *
from network import *
from sklearn.preprocessing import LabelEncoder
from get_threshold import threshold_ver2, threshold_ver2_both, threshold_ver1, Refiner
from crop_image import crop_image


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
test_path = f'../data/test' 
test_folders = natsorted(glob(test_path + '/*'))

args = EasyDict({'encoder_name':'regnety_040',
                        'drop_path_rate':0,
                        })

load_pretrain = True        # Use Pretrained weights
ensemble_test = True        # Ensemble or Single
refine = True              # Use Refiner (Rule-base)

## Pretrained weight download from github
# os.makedirs('./results/', exist_ok=True)
# !wget -i https://raw.githubusercontent.com/wooseok-shin/Egovision-1st-place-solution/main/load_pretrained.txt -P results
if load_pretrain:  # Github로부터 Pretrained Weight Load
    model_path0 = './results/0Fold_model.pth' # fold0
    model_path1 = './results/1Fold_model.pth' # fold1
    model_path2 = './results/2Fold_model.pth' # fold2
    model_path3 = './results/3Fold_model.pth' # fold3
    model_path4 = './results/4Fold_model.pth' # fold4

else:  # 학습한 모델 Weight Load
    model_path0 = './results/000/best_model.pth' # fold0
    model_path1 = './results/001/best_model.pth' # fold1
    model_path2 = './results/002/best_model.pth' # fold2
    model_path3 = './results/003/best_model.pth' # fold3
    model_path4 = './results/004/best_model.pth' # fold4


# 5Fold Ensemble
if ensemble_test:
    model0 = Pose_Network(args).to(device)
    model0.load_state_dict(torch.load(model_path0)['state_dict'])
    model0.eval()

    model1 = Pose_Network(args).to(device)
    model1.load_state_dict(torch.load(model_path1)['state_dict'])
    model1.eval()

    model2 = Pose_Network(args).to(device)
    model2.load_state_dict(torch.load(model_path2)['state_dict'])
    model2.eval()

    model3 = Pose_Network(args).to(device)
    model3.load_state_dict(torch.load(model_path3)['state_dict'])
    model3.eval()

    model4 = Pose_Network(args).to(device)
    model4.load_state_dict(torch.load(model_path4)['state_dict'])
    model4.eval()

    model_list = [model0, model1, model2, model3, model4]

else:  # Single Best Model (Using the pretrained weight)
    model_path = './results/single_best_model.pth'
    single_best = Pose_Network(args).to(device)
    single_best.load_state_dict(torch.load(model_path)['state_dict'])
    single_best.eval()
    model_list = [single_best]


img_size = 288
transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((img_size, img_size)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        ])

sub = pd.read_csv('../data/sample_submission.csv')
df_info = pd.read_csv('../data/hand_gesture_pose.csv')
le = LabelEncoder()
le.fit(df_info['pose_id'])
trans = le.transform

# Class Mapping dict
ver1_list = trans([0, 42, 10, 67, 100, 142, 110, 167])   
ver2_list = trans([146, 163, 171, 191])
replace_dict = {146:163, 171:191, 0:42, 10:67, 100:142, 110:167}
replace_dict = dict([trans(x) for x in list(replace_dict.items())])   # Mapping (Origin:0~195 to 0~156)

total_list = np.concatenate([ver1_list, ver2_list]).tolist()


for i, test_folder in tqdm(enumerate(test_folders)):
    dir = os.path.dirname(test_folder)
    folder_num = os.path.basename(test_folder)
    json_path = opj(dir, folder_num, folder_num+'.json')
    js = json.load(open(json_path))
    keypoints = js['annotations']  # 해당 이미지에 해당하는 Keypoints
    images_list = natsorted(glob(test_folder + '/*.png'))
    images = []
    for _, (point, image_name) in enumerate(zip(keypoints, images_list)):
        croped_image = crop_image(image_name, point, margin=100)
        image = transform(croped_image)
        images.append(image)

    images = torch.stack(images).to(device)
    ensemble = np.zeros((157,), dtype=np.float32)
    for model in model_list:
        preds = model(images)
        preds = torch.softmax(preds, dim=1)
        preds = torch.mean(preds, dim=0).detach().cpu().numpy()    # shape:(157,)
        ensemble += preds
    preds = ensemble / len(model_list)
    pred_class = preds.argmax().item()
    if refine and (pred_class in total_list):
        idx = list(replace_dict.keys()).index(pred_class) if pred_class in replace_dict.keys() else list(replace_dict.values()).index(pred_class)
        cand1, cand2 = list(replace_dict.items())[idx]

        if pred_class in ver1_list:
            query_value = Refiner(keypoints, ver=1)
            answer = cand1 if query_value < threshold_ver1 else cand2

        elif pred_class in ver2_list:
            query_value = Refiner(keypoints, ver=2)
            answer = cand1 if query_value < threshold_ver2_both else cand2

        preds[answer] = 1
        preds = np.where(preds != 1, 0, preds)  # Refiner를 통해 나온 class를 제외한 나머지의 확률값은 모두 0으로 변환

    sub.iloc[i, 1:] = preds.astype(float)

sub.to_csv('./results/submission_train_add_ensemble_rule.csv',index=False)
# %%
