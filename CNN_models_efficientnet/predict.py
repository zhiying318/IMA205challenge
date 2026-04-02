# predict.py
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import pandas as pd

from PIL import Image
from torchvision import transforms
from model import get_model
from tqdm import tqdm
import torchvision.transforms.functional as F
import timm

# # 1. 配置
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# BATCH_SIZE = 64
# IMG_SIZE = 256
# MODEL_PATH = "best_wbc_model.pth"
# TEST_METADATA = "test_metadata.csv"
# TEST_IMG_DIR = "./data_cropped/test"  # 预处理后的测试集路径

# # 2. 这里的类别顺序必须和训练时 ImageFolder 的顺序完全一致
# # 建议在 train.py 运行后记录下 train_loader.dataset.classes 的输出
# # 下面是一个假设的列表，请根据你文件夹生成的顺序修改
# # 根据你的打印结果填写的映射列表
# CLASS_NAMES = [
#     'BA',   # 0
#     'BL',   # 1
#     'BNE',  # 2
#     'EO',   # 3
#     'LY',   # 4
#     'MMY',  # 5
#     'MO',   # 6
#     'MY',   # 7
#     'PC',   # 8
#     'PLY',  # 9
#     'PMY',  # 10
#     'SNE',  # 11
#     'VLY'   # 12
# ]

# # 3. 数据转换 (必须与训练时的 val_transform 一致)
# test_transform = transforms.Compose([
#     transforms.Resize((IMG_SIZE, IMG_SIZE)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])

# def predict():
#     # 加载模型
#     # model = get_model(num_classes=len(CLASS_NAMES), model_name='resnet18')
#     # below are only for resnet50 with dropout, if you used resnet18 without dropout, please use the above line instead
#     model = get_model(num_classes=len(CLASS_NAMES), model_name='resnet50')
    
#     model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
#     model.to(DEVICE)
#     model.eval()

#     # 读取测试集元数据
#     test_df = pd.read_csv(TEST_METADATA)
#     predictions = []

#     print("Starting Inference...")
#     with torch.no_grad():
#         for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
#             img_id = row['ID']
#             img_path = os.path.join(TEST_IMG_DIR, img_id)
            
#             # 加载并转换图像
#             try:
#                 image = Image.open(img_path).convert("RGB")
#                 input_tensor = test_transform(image).unsqueeze(0).to(DEVICE)
                
#                 # 推理
#                 output = model(input_tensor)
#                 pred_idx = torch.argmax(output, 1).item()
#                 predictions.append(CLASS_NAMES[pred_idx])
#             except Exception as e:
#                 print(f"Error processing {img_id}: {e}")
#                 predictions.append("SNE") # 如果出错，默认填出现频率最高的类

#     # 保存结果
#     test_df['label'] = predictions
#     test_df[['ID', 'label']].to_csv("submission_cnn_v0.csv", index=False)
#     print("Submission saved as submission_cnn_v0.csv")

# if __name__ == "__main__":
#     predict()


# 配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "best_wbc_model_efficientnet_b3_focalloss_RandAug_Mixup.pth"
TEST_METADATA = "test_metadata.csv"
TEST_IMG_DIR = "./data_cropped/test_eff"
IMG_SIZE = 300 # 保持与训练一致 or 224

CLASS_NAMES = ['BA', 'BL', 'BNE', 'EO', 'LY', 'MMY', 'MO', 'MY', 'PC', 'PLY', 'PMY', 'SNE', 'VLY']

# 基础转换
test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict():
    # 1. 加载模型
    model = get_model(num_classes=len(CLASS_NAMES), model_name='efficientnet_b3')
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    test_df = pd.read_csv(TEST_METADATA)
    predictions = []

    print(f"Starting TTA Inference on {DEVICE}...")
    
    with torch.no_grad():
        for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
            img_path = os.path.join(TEST_IMG_DIR, row['ID'])
            try:
                image = Image.open(img_path).convert("RGB")
                
                # --- 8视图 TTA：4个旋转角度 × 是否水平翻转 ---
                tta_logits = []
                for angle in [0, 90, 180, 270]:
                    img_rot = F.rotate(image, angle)
                    # 原始旋转
                    tta_logits.append(model(test_transform(img_rot).unsqueeze(0).to(DEVICE)))
                    # 旋转后翻转
                    img_flip = F.hflip(img_rot)
                    tta_logits.append(model(test_transform(img_flip).unsqueeze(0).to(DEVICE)))
                
                # 对所有 Logits 求平均 (而不是对 Softmax 后的概率)
                all_logits = torch.cat(tta_logits, dim=0) # (8, 13)
                avg_logits = torch.mean(all_logits, dim=0, keepdim=True) # (1, 13)
                
                pred_idx = torch.argmax(avg_logits, dim=1).item()
                predictions.append(CLASS_NAMES[pred_idx])
                
            except Exception as e:
                predictions.append("SNE")

    # 保存
    test_df['label'] = predictions
    test_df[['ID', 'label']].to_csv("submission_efficientnet_b3_focalloss_RandAug_Mixup_tta.csv", index=False)
    print("Inference Complete. TTA submission saved.")

if __name__ == "__main__":
    predict()