# predict_ensemble.py
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn.functional as F
import pandas as pd
import timm
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF
from model import get_model
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ['BA', 'BL', 'BNE', 'EO', 'LY', 'MMY', 'MO', 'MY', 'PC', 'PLY', 'PMY', 'SNE', 'VLY']
TEST_METADATA = "test_metadata.csv"
TEST_IMG_DIR = "./data_cropped/test_eff"  # 两个模型共用同一份裁剪好的测试集

def get_transform(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

# -------------------------------------------------------
# 单张图片做8视图TTA，返回平均后的概率向量
# -------------------------------------------------------
def tta_predict_single(model, image, transform):
    """
    image: PIL Image
    返回: shape (13,) 的概率tensor，已softmax
    """
    tta_logits = []
    for angle in [0, 90, 180, 270]:
        img_rot = TF.rotate(image, angle)
        # 原始旋转
        logit = model(transform(img_rot).unsqueeze(0).to(DEVICE))
        tta_logits.append(logit)
        # 水平翻转
        logit_flip = model(transform(TF.hflip(img_rot)).unsqueeze(0).to(DEVICE))
        tta_logits.append(logit_flip)
    
    avg_logits = torch.mean(torch.cat(tta_logits, dim=0), dim=0)  # (13,)
    return F.softmax(avg_logits, dim=0)  # 转成概率，方便跨模型平均


def load_models():
    models_cfg = [
        # EfficientNet-B3
        (
            get_model(num_classes=13, model_name='efficientnet_b3'),
            "best_wbc_model_efficientnet_b3_focalloss.pth",
            300
        ),
        # ConvNeXt-Large
        (
            timm.create_model('convnext_large', pretrained=False, num_classes=13),
            "best_wbc_model_convnext_large.pth",
            224
        ),
    ]
    
    loaded = []
    for model, ckpt_path, img_size in models_cfg:
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        loaded.append((model, get_transform(img_size)))
        print(f"Loaded: {ckpt_path}")
    
    return loaded


# -------------------------------------------------------
# 主推理函数
# -------------------------------------------------------
def predict_ensemble():
    models = load_models()
    test_df = pd.read_csv(TEST_METADATA)
    predictions = []

    with torch.no_grad():
        for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
            img_path = os.path.join(TEST_IMG_DIR, row['ID'])
            try:
                image = Image.open(img_path).convert("RGB")

                # 每个模型各自TTA，得到概率向量
                model_probs = []
                for model, transform in models:
                    prob = tta_predict_single(model, image, transform)  # (13,)
                    model_probs.append(prob)
                
                # 跨模型平均概率
                avg_prob = torch.stack(model_probs).mean(dim=0)  # (13,)
                pred_idx = avg_prob.argmax().item()
                predictions.append(CLASS_NAMES[pred_idx])

            except Exception as e:
                print(f"Error: {row['ID']} -> {e}")
                predictions.append("SNE")

    test_df['label'] = predictions
    out_name = f"submission_ensemble_{len(models)}model_tta8.csv"
    test_df[['ID', 'label']].to_csv(out_name, index=False)
    print(f"✅ 完成，已保存: {out_name}")


if __name__ == "__main__":
    predict_ensemble()


# ```

# ---

# ## 结构逻辑一图说明
# ```
# 每张测试图片
#     │
#     ├─ EfficientNet-B3 (300×300)
#     │       ├─ 0°原图  → logit
#     │       ├─ 0°翻转  → logit
#     │       ├─ 90°     → logit
#     │       ├─ 90°翻转 → logit
#     │       ├─ ...共8个
#     │       └─ mean → softmax → prob向量 (13,)
#     │
#     ├─ ConvNeXt-Large (224×224)
#     │       └─ 同上 8视图TTA → prob向量 (13,)
#     │
#     └─ 两个prob向量平均 → argmax → 最终预测