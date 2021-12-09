from PIL import Image
import numpy as np

def crop_image(imges, point, margin=100):
    image = np.array(Image.open(imges).convert('RGB'))
    point = point['data']
    max_point = np.max(np.array(point), axis=0).astype(int) + margin
    min_point = np.min(np.array(point), axis=0).astype(int) - margin
    max_point = max_point[:-1] # remove Z order
    min_point = min_point[:-1] # remove Z order

    max_x, max_y = max_point
    min_x, min_y = min_point
    max_y += margin  # 손목까지
    
    # 데이터 포인트의 크기가 원 이미지를 넘어서는 경우를 방지
    max_x = max_x if max_x < 1920 else 1920
    max_y = max_y if max_y < 1080 else 1080
    min_x = min_x if min_x > 0 else 0
    min_y = min_y if min_y > 0 else 0
    
    crop_image = image[min_y:max_y, min_x:max_x]

    return crop_image