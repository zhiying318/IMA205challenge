# preprocess.py
import os
import cv2
import pandas as pd
from tqdm import tqdm
from PIL import Image
import numpy as np


# 导入你之前的裁剪函数
def apply_wbc_crop_robust(img_rgb, fixed_size=300): # efficientnet 版本用 300x300 的输入，所以裁剪时直接裁成 300x300，resnet 版本则裁成 280x280
    h, w = img_rgb.shape[:2]
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    mask = ((img_hsv[:, :, 1] > 50) & (img_hsv[:, :, 2] < 190)).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1: return img_rgb
    img_center = np.array([w / 2, h / 2])
    best_idx = -1
    min_dist = float('inf')
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] < 400: continue
        dist = np.linalg.norm(centroids[i] - img_center)
        if dist < min_dist:
            min_dist = dist; best_idx = i
    if best_idx == -1: return img_rgb
    cx, cy = int(centroids[best_idx][0]), int(centroids[best_idx][1])
    half_s = fixed_size // 2
    x1, x2, y1, y2 = cx-half_s, cx+half_s, cy-half_s, cy+half_s
    # 越界处理逻辑 (简略版，建议用你之前的完整版)
    x1, y1 = max(0, x1), max(0, y1)
    return img_rgb[y1:y2, x1:x2]

def prepare_dataset(csv_path, src_dir, dst_dir, is_test=False):
    df = pd.read_csv(csv_path)
    os.makedirs(dst_dir, exist_ok=True)
    for _, row in tqdm(df.iterrows(), total=len(df)):
        img_path = os.path.join(src_dir, row['ID'])
        img = np.array(Image.open(img_path).convert("RGB"))
        cropped = apply_wbc_crop_robust(img)
        
        if not is_test:
            # 训练集按类别存入文件夹：dst/CLASS/ID.jpg
            class_dir = os.path.join(dst_dir, str(row['label']))
            os.makedirs(class_dir, exist_ok=True)
            save_path = os.path.join(class_dir, row['ID'])
        else:
            # 测试集直接存：dst/ID.jpg
            save_path = os.path.join(dst_dir, row['ID'])
            
        Image.fromarray(cropped).save(save_path, quality=95)

if __name__ == "__main__":
    prepare_dataset("train_metadata.csv", "./train", "./data_cropped/train_eff")
    prepare_dataset("test_metadata.csv", "./test", "./data_cropped/test_eff", is_test=True)