import os
import time
import logging
import pandas as pd
import torch
import torch.nn as nn
import torch_optimizer as optim
from os.path import join as opj
from tqdm import tqdm
from ptflops import get_model_complexity_info
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from torch.cuda.amp import autocast, grad_scaler
from natsort import natsorted
from glob import glob
from utils import get_root_logger, WarmUpLR, AvgMeter
from dataloader import *
from network import *

import warnings
warnings.filterwarnings('ignore')

class Trainer():
    def __init__(self, args, save_path):
        '''
        args: arguments
        save_path: Model 가중치 저장 경로
        '''
        super(Trainer, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Logging
        log_file = os.path.join(save_path, 'log.log')
        self.logger = get_root_logger(logger_name='IR', log_level=logging.INFO, log_file=log_file)
        self.logger.info(args)
        self.logger.info(args.tag)

        # Train, Valid Set load
        ############################################################################
        # df_train = pd.read_csv(opj(args.data_path, 'df_train.csv'))
        df_train = pd.read_csv(opj(args.data_path, 'df_train_add.csv'))
        df_info = pd.read_csv(opj(args.data_path, 'hand_gesture_pose.csv'))

        df_train = df_train.merge(df_info[['pose_id', 'gesture_type', 'hand_type']], \
                                how='left', left_on='answer', right_on='pose_id')
        df_train['groups'] = df_train['train_path'].apply(lambda x:x.split('/')[3])
        df_train.loc[:,:] = natsorted(df_train.values)
        drop_idx = [9] #+ df_train[df_train['groups'].isin(['596'])].index.tolist()[3:8]
        df_train = df_train.drop(drop_idx).reset_index(drop=True)  
        le = LabelEncoder()
        df_train['answer'] = le.fit_transform(df_train['answer'])
        
        # Split Fold
        # kf = StratifiedGroupKFold(n_splits=args.Kfold)
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
        for fold, (train_idx, val_idx) in enumerate(kf.split(df_train, y=df_train['answer'])):
        # for fold, (train_idx, val_idx) in enumerate(kf.split(df_train, y=df_train['answer'], groups=df_train['groups'])):
            df_train.loc[val_idx, 'fold'] = fold

        # df_train.to_csv('df_train_gk_fold.csv' , index=False)
        df_val = df_train[df_train['fold'] == args.fold].reset_index(drop=True)
        df_train = df_train[df_train['fold'] != args.fold].reset_index(drop=True)
        
        # Augmentation
        self.train_transform = get_train_augmentation(img_size=args.img_size, ver=args.aug_ver)
        self.test_transform = get_train_augmentation(img_size=args.img_size, ver=1)

        # Flip Augmentation을 위한 Mapping dataframe
        df_info = pd.read_csv('../data/hand_gesture_pose.csv')
        df_info = df_info[df_info['hand_type'] != 'both']
        # drop idx, 동일한 약속, gesture_type, hand_type인데 다른 클래스인 경우 존재 -> 약속 1과 2로 이름을 나누어줌.
        df_info.loc[[105, 128], 'pose_name'] = '약속 1'  # idx: (105, 128)
        df_info.loc[[101, 124], 'pose_name'] = '약속 2'  # idx: (101, 124)

        # drop 41 idx, 동일한 약속, my hand, right class가 49와 54로 두 개있어 Mapping df만들 때 문제가 발생하여 미리 49번 클래스 처리
        df_info = df_info.drop(41)

        # Make a mapping dataframe
        df_info = df_info.groupby(['pose_name', 'view_type', 'gesture_type', 'hand_type']).sum().unstack().reset_index().dropna(axis=0) # df_infoture_type??
        df_info['left'] = df_info.pose_id.left.apply(int) 
        df_info['right'] = df_info.pose_id.right.apply(int)
        df_flip_info = df_info.drop('pose_id', axis=1).droplevel('hand_type', axis=1).reset_index(drop=True)
        print('Mapping dataframe Length', df_flip_info.shape)
        
        # TrainLoader
        self.train_loader = get_loader(df_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, transform=self.train_transform, 
                                       df_flip_info=df_flip_info, flipaug_ratio=args.flipaug_ratio, label_encoder=le, margin=args.margin, random_margin=args.random_margin)
        self.val_loader = get_loader(df_val, batch_size=args.batch_size, shuffle=False,
                                       num_workers=args.num_workers, transform=self.test_transform)

        # Network
        self.model = Pose_Network(args).to(self.device)
        macs, params = get_model_complexity_info(self.model, (3, args.img_size, args.img_size), as_strings=True,
                                                 print_per_layer_stat=False, verbose=False)
        self.logger.info('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        self.logger.info('{:<30}  {:<8}'.format('Number of parameters: ', params))

        # Loss
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer & Scheduler
        self.optimizer = optim.Lamb(self.model.parameters(), lr=args.initial_lr, weight_decay=args.weight_decay)
        
        iter_per_epoch = len(self.train_loader)
        self.warmup_scheduler = WarmUpLR(self.optimizer, iter_per_epoch * args.warm_epoch)

        if args.scheduler == 'cos':
            tmax = args.tmax # half-cycle 
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max = tmax, eta_min=args.min_lr, verbose=True)
        elif args.scheduler == 'cycle':
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=args.max_lr, steps_per_epoch=iter_per_epoch, epochs=args.epochs)

        
        if args.multi_gpu:
            self.model = nn.DataParallel(self.model).to(self.device)

        # Train / Validate
        best_loss = np.inf
        best_acc = 0
        best_epoch = 0
        early_stopping = 0
        start = time.time()
        for epoch in range(1, args.epochs+1):
            self.epoch = epoch

            if args.scheduler == 'cos':
                if epoch > args.warm_epoch:
                    self.scheduler.step()

            # Training
            train_loss, train_acc = self.training(args)

            # Model weight in Multi_GPU or Single GPU
            state_dict= self.model.module.state_dict() if args.multi_gpu else self.model.state_dict()

            # Validation
            val_loss, val_acc = self.validate()

            # Save models
            if val_loss < best_loss:
                early_stopping = 0
                best_epoch = epoch
                best_loss = val_loss
                best_acc = val_acc

                torch.save({'epoch':epoch,
                            'state_dict':state_dict,
                            'optimizer': self.optimizer.state_dict(),
                            'scheduler': self.scheduler.state_dict(),
                    }, os.path.join(save_path, 'best_model.pth'))
                self.logger.info(f'-----------------SAVE:{best_epoch}epoch----------------')
            else:
                early_stopping += 1

            # Early Stopping
            if early_stopping == args.patience:
                break

        self.logger.info(f'\nBest Val Epoch:{best_epoch} | Val Loss:{best_loss:.4f} | Val Acc:{best_acc:.4f}')
        end = time.time()
        self.logger.info(f'Total Process time:{(end - start) / 60:.3f}Minute')

    # Training
    def training(self, args):
        self.model.train()
        train_loss = AvgMeter()
        train_acc = 0

        scaler = grad_scaler.GradScaler()
        for i, (images, targets) in enumerate(tqdm(self.train_loader)):
            images = torch.tensor(images, device=self.device, dtype=torch.float32)
            targets = torch.tensor(targets, device=self.device, dtype=torch.long)
            
            if self.epoch <= args.warm_epoch:
                self.warmup_scheduler.step()

            self.model.zero_grad(set_to_none=True)
            if args.amp:
                with autocast():
                    preds = self.model(images)
                    loss = self.criterion(preds, targets)
                scaler.scale(loss).backward()

                # Gradient Clipping
                if args.clipping is not None:
                    scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), args.clipping)

                scaler.step(self.optimizer)
                scaler.update()

            else:
                preds = self.model(images)
                loss = self.criterion(preds, targets)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), args.clipping)
                self.optimizer.step()

            if args.scheduler == 'cycle':
                if self.epoch > args.warm_epoch:
                    self.scheduler.step()

            # Metric
            train_acc += (preds.argmax(dim=1) == targets).sum().item()
            # log
            train_loss.update(loss.item(), n=images.size(0))
            
        train_acc /= len(self.train_loader.dataset)

        self.logger.info(f'Epoch:[{self.epoch:03d}/{args.epochs:03d}]')
        self.logger.info(f'Train Loss:{train_loss.avg:.3f} | Acc:{train_acc:.4f}')
        return train_loss.avg, train_acc
            
    # Validation or Dev
    def validate(self):
        self.model.eval()
        with torch.no_grad():
            val_loss = AvgMeter()
            val_acc = 0

            for _, (images, targets) in enumerate(self.val_loader):
                images = torch.tensor(images, device=self.device, dtype=torch.float32)
                targets = torch.tensor(targets, device=self.device, dtype=torch.long)

                preds = self.model(images)
                loss = self.criterion(preds, targets)

                # Metric
                val_acc += (preds.argmax(dim=1) == targets).sum().item()
                # log
                val_loss.update(loss.item(), n=images.size(0))
            val_acc /= len(self.val_loader.dataset)

            self.logger.info(f'Valid Loss:{val_loss.avg:.3f} | Acc:{val_acc:.4f}')
        return val_loss.avg, val_acc