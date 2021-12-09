import argparse

def getConfig():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_num', default='1', type=str, help='experiment_number')
    parser.add_argument('--experiment', default='Base', type=str)
    parser.add_argument('--tag', default='Default', type=str, help='tag')

    # Path settings
    parser.add_argument('--data_path', type=str, default='../data')
    parser.add_argument('--fold', type=int, default=0, help='Validation Fold')
    parser.add_argument('--Kfold', type=int, default=5, help='Number of Split Folds')
    parser.add_argument('--model_path', type=str, default='results/')

    # Model parameter settings
    parser.add_argument('--encoder_name', type=str, default='regnety_040')
    parser.add_argument('--drop_path_rate', type=float, default=0.2)

    # Training parameter settings
    ## Base Parameter
    parser.add_argument('--img_size', type=int, default=288)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--optimizer', type=str, default='Lamb')
    parser.add_argument('--initial_lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    
    ## Augmentation
    parser.add_argument('--aug_ver', type=int, default=1)
    parser.add_argument('--flipaug_ratio', type=float, default=0.3)
    parser.add_argument('--margin', type=int, default=50)
    parser.add_argument('--random_margin', type=bool, default=True)

    ## Scheduler
    parser.add_argument('--scheduler', type=str, default='cos')
    parser.add_argument('--warm_epoch', type=int, default=5)  # WarmUp Scheduler
    parser.add_argument('--freeze_epoch', type=int, default=0)
    ### Cosine Annealing
    parser.add_argument('--min_lr', type=float, default=5e-6)
    parser.add_argument('--tmax', type=int, default = 145)
    ### MultiStepLR
    parser.add_argument('--milestone', type=int, nargs='*', default=[50])
    parser.add_argument('--lr_factor', type=float, default=0.1)
	### OnecycleLR
    parser.add_argument('--max_lr', type=float, default=1e-3)

    ## etc.
    parser.add_argument('--patience', type=int, default=50, help='Early Stopping')
    parser.add_argument('--clipping', type=float, default=None, help='Gradient clipping')

    # Hardware settings
    parser.add_argument('--amp', default=True)
    parser.add_argument('--multi_gpu', type=bool, default=False)
    parser.add_argument('--logging', type=bool, default=False)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = getConfig()
    args = vars(args)
    print(args)
