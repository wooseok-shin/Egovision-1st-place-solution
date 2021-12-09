python main.py --batch_size=16 --encoder_name=regnety_040 --flipaug_ratio=0.3\
    			--img_size=288 --aug_ver=2 --scheduler=cycle --margin=50 --fold=0\
                --weight_decay=1e-3 --initial_lr=5e-6\
                --max_lr=1e-3 --epochs=50 --warm_epoch=5 --exp_num=0

python main.py --batch_size=16 --encoder_name=regnety_040 --flipaug_ratio=0.3\
    			--img_size=288 --aug_ver=2 --scheduler=cycle --margin=50 --fold=1\
                --weight_decay=1e-3 --initial_lr=5e-6\
                --max_lr=1e-3 --epochs=50 --warm_epoch=5 --exp_num=1

python main.py --batch_size=16 --encoder_name=regnety_040 --flipaug_ratio=0.3\
    			--img_size=288 --aug_ver=2 --scheduler=cycle --margin=50 --fold=2\
                --weight_decay=1e-3 --initial_lr=5e-6\
                --max_lr=1e-3 --epochs=50 --warm_epoch=5 --exp_num=2

python main.py --batch_size=16 --encoder_name=regnety_040 --flipaug_ratio=0.3\
    			--img_size=288 --aug_ver=2 --scheduler=cycle --margin=50 --fold=3\
                --weight_decay=1e-3 --initial_lr=5e-6\
                --max_lr=1e-3 --epochs=50 --warm_epoch=5 --exp_num=3

python main.py --batch_size=16 --encoder_name=regnety_040 --flipaug_ratio=0.3\
    			--img_size=288 --aug_ver=2 --scheduler=cycle --margin=50 --fold=4\
                --weight_decay=1e-3 --initial_lr=5e-6\
                --max_lr=1e-3 --epochs=50 --warm_epoch=5 --exp_num=4
