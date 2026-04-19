# inference_improved.py
# 改进点：
# 1. 按验证集 F1 加权 ensemble（而非简单平均）
# 2. SNE/BNE 后处理校准（解决最大混淆源）
# 3. 支持 temperature scaling
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn.functional as F
import pandas as pd
import timm
import numpy as np
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF
from model import get_model
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ['BA', 'BL', 'BNE', 'EO', 'LY', 'MMY', 'MO', 'MY', 'PC', 'PLY', 'PMY', 'SNE', 'VLY']
TEST_METADATA = "test_metadata.csv"
TEST_IMG_DIR = "./data_cropped/test_eff"

# ============================================================
# 各 fold 验证集 Macro-F1（用于加权）
# ============================================================
EFFNET_F1 = [0.6871, 0.6414, 0.6295, 0.6109, 0.6419]
CONVNEXT_F1 = [0.6253, 0.6656, 0.6505, 0.6288, 0.6910]

def softmax_weights(f1_scores, temperature=2.0):
    """将 F1 分数转为 softmax 权重，temperature 越小越集中在最好的模型"""
    f1 = np.array(f1_scores)
    exp_f1 = np.exp(f1 / temperature)
    return exp_f1 / exp_f1.sum()

def get_transform(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def tta_predict_single(model, image, transform):
    """8视图 TTA，返回 softmax 概率向量 (13,)"""
    tta_logits = []
    for angle in [0, 90, 180, 270]:
        img_rot = TF.rotate(image, angle)
        logit = model(transform(img_rot).unsqueeze(0).to(DEVICE))
        tta_logits.append(logit)
        logit_flip = model(transform(TF.hflip(img_rot)).unsqueeze(0).to(DEVICE))
        tta_logits.append(logit_flip)
    
    avg_logits = torch.mean(torch.cat(tta_logits, dim=0), dim=0)
    return F.softmax(avg_logits, dim=0)


def load_models():
    """加载所有模型并返回 (model, transform, weight) 三元组"""
    all_f1 = EFFNET_F1 + CONVNEXT_F1
    model_weights = softmax_weights(all_f1, temperature=0.5)
    # temperature=0.05 会让权重几乎全集中在最好的几个模型上
    # 如果你想让所有模型都有贡献，用 temperature=2.0
    # 建议先用 temperature=0.5 试试
    
    print("=" * 50)
    print("模型加权方案:")
    
    models_cfg = []
    for fold in range(5):
        models_cfg.append((
            get_model(num_classes=13, model_name='efficientnet_b3'),
            f"best_efficientnet_fold{fold}.pth",
            300,
            f"EfficientNet fold{fold} (F1={EFFNET_F1[fold]:.4f})"
        ))
    for fold in range(5):
        models_cfg.append((
            timm.create_model('convnext_large', pretrained=False, num_classes=13),
            f"best_convnext_fold{fold}.pth",
            224,
            f"ConvNeXt fold{fold} (F1={CONVNEXT_F1[fold]:.4f})"
        ))
    
    loaded = []
    for i, (model, ckpt_path, img_size, desc) in enumerate(models_cfg):
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        w = model_weights[i]
        loaded.append((model, get_transform(img_size), w))
        print(f"  {desc} -> weight={w:.4f}")
    
    print("=" * 50)
    return loaded


def post_process(avg_prob, class_names=CLASS_NAMES):
    """
    后处理规则，解决已知的高混淆对。
    
    核心思路：利用先验知识校准。
    - SNE 有 13015 个样本，BNE 只有 391 个
    - 当模型对 BNE 没有很高置信度时，SNE 更可能是正确答案
    """
    prob = avg_prob.clone()
    
    bne_idx = 2   # BNE
    sne_idx = 11  # SNE
    vly_idx = 12  # VLY
    ly_idx = 4    # LY
    mmy_idx = 5   # MMY
    my_idx = 7    # MY（MMY最容易被误判为MY）
    
    pred_idx = prob.argmax().item()
    
    # 规则1: BNE vs SNE 校准
    # 如果预测为 BNE，但 BNE 概率不够高（<阈值），且 SNE 概率也不低，则改判 SNE
    if pred_idx == bne_idx:
        bne_prob = prob[bne_idx].item()
        sne_prob = prob[sne_idx].item()
        # SNE 样本是 BNE 的 33 倍，所以 BNE 需要更高置信度才可信
        if bne_prob < 0.45 and sne_prob > 0.15:
            prob[sne_idx] *= 1.5  # 提升 SNE 权重
            pred_idx = prob.argmax().item()
    
    # 规则2: VLY vs LY 校准
    # VLY 只有 366 个，LY 有 8101 个
    if pred_idx == vly_idx:
        vly_prob = prob[vly_idx].item()
        ly_prob = prob[ly_idx].item()
        if vly_prob < 0.40 and ly_prob > 0.20:
            prob[ly_idx] *= 1.3
            pred_idx = prob.argmax().item()

    # 规则3: MMY vs MY 校准
    # MMY 只有 122 个，MY 有 1177 个
    if pred_idx == mmy_idx:
        mmy_prob = prob[mmy_idx].item()
        my_prob = prob[my_idx].item()
        if mmy_prob < 0.40 and my_prob > 0.25:
            prob[my_idx] *= 1.2
            pred_idx = prob.argmax().item()
    
    return pred_idx


def predict_ensemble():
    models = load_models()
    test_df = pd.read_csv(TEST_METADATA)
    predictions = []
    predictions_no_pp = []  # 不带后处理的预测，用于对比

    with torch.no_grad():
        for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
            img_path = os.path.join(TEST_IMG_DIR, row['ID'])
            try:
                image = Image.open(img_path).convert("RGB")

                # 加权 ensemble
                weighted_prob = torch.zeros(13).to(DEVICE)
                for model, transform, weight in models:
                    prob = tta_predict_single(model, image, transform)
                    weighted_prob += weight * prob
                
                # 不带后处理
                pred_no_pp = weighted_prob.argmax().item()
                predictions_no_pp.append(CLASS_NAMES[pred_no_pp])
                
                # 带后处理
                pred_idx = post_process(weighted_prob)
                predictions.append(CLASS_NAMES[pred_idx])

            except Exception as e:
                print(f"Error: {row['ID']} -> {e}")
                predictions.append("SNE")
                predictions_no_pp.append("SNE")

    # 保存两个版本对比
    test_df['label'] = predictions
    out_name = "submission_ensemble/submission_weighted_ensemble_pp_05_1904.csv"
    test_df[['ID', 'label']].to_csv(out_name, index=False)
    
    test_df['label'] = predictions_no_pp
    out_name2 = "submission_ensemble/submission_weighted_ensemble_no_pp_05_1904.csv"
    test_df[['ID', 'label']].to_csv(out_name2, index=False)
    
    # 统计后处理改变了多少预测
    changed = sum(1 for a, b in zip(predictions, predictions_no_pp) if a != b)
    print(f"\n✅ 完成!")
    print(f"  加权 ensemble + 后处理: {out_name}")
    print(f"  加权 ensemble 无后处理: {out_name2}")
    print(f"  后处理改变了 {changed}/{len(predictions)} 个预测")


if __name__ == "__main__":
    predict_ensemble()