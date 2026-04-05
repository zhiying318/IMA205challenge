# train.py
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast, GradScaler
from model import get_model
from dataset import get_loaders, get_loaders_fold
from sklearn.metrics import f1_score, classification_report
import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import timm
import numpy as np
import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def log_message(message, log_file="training_convnext_2_kfold.log"):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_msg = f"[{timestamp}] {message}"
    print(formatted_msg) # 打印到 tmux 屏幕
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(formatted_msg + "\n") # 写入文件，方便外面 tail -f 查看

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, targets,
            weight=self.alpha,
            label_smoothing=self.label_smoothing,
            reduction='none'
        )
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


def mixup_data(x, y, alpha=0.2):
    '''返回混合后的输入、两份原始标签以及混合比例 lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    '''混合 Loss 的计算函数'''
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# 1. 计算类别权重 (根据你提供的样本数)(根据你提供的顺序：BA 到 VLY)
# 顺序必须严格对应：['BA', 'BL', 'BNE', 'EO', 'LY', 'MMY', 'MO', 'MY', 'PC', 'PLY', 'PMY', 'SNE', 'VLY']
# counts = [415, 2012, 391, 861, 8101, 360, 2746, 441, 68, 11, 114, 13015, 366] 以下权重太激进，反而导致训练效果不好
# weights = 1.0 / torch.tensor(counts, dtype=torch.float)
# weights[9] = weights[9] * 5.0  # 给 PLY 5倍补偿
# weights[8] = weights[8] * 2.0  # 给 PC 2倍补偿
# weights[2] = weights[2] * 2.0  # 给 BNE 2倍补偿
counts = torch.tensor([415, 2012, 391, 861, 8101, 360, 2746, 441, 68, 11, 114, 13015, 366], dtype=torch.float)
weights = 1.0 / torch.sqrt(counts) # 使用平方根，平滑权重梯度
weights = weights / weights.sum() * len(counts) # 归一化，防止 Loss 太小
weights = weights.to(device)

N_FOLDS = 2
def train(use_convnext=False):
    if use_convnext:
        arch = "convnext"
        img_size = 224          # ConvNeXt标准输入尺寸
        batch_size = 64         # ConvNeXt-Large显存占用大，batch调小
        model_save_path = "best_wbc_model_convnext_large_focalloss_RandAug.pth"
        cm_save_dir = "./CNN_models_convnext/confusion_matrix_focalloss_RandAug"
    else:
        arch = "efficientnet"
        img_size = 300          # EfficientNet-B3
        batch_size = 64
        model_save_path = "best_wbc_model_efficientnet_b3_focalloss_RandAug.pth"
        cm_save_dir = "./CNN_models_efficientnet/confusion_matrix_focalloss_RandAug"
    
    os.makedirs(cm_save_dir, exist_ok=True)


    for fold in range(N_FOLDS):
        log_message(f"\n{'='*50}")
        log_message(f"  {arch.upper()} — Fold {fold+1}/{N_FOLDS}")
        log_message(f"{'='*50}")

        model_save_path = f"best_{arch}_fold{fold}.pth"
        cm_save_dir = f"./CNN_models_{arch}/confusion_matrix_{arch}/fold{fold}"
        os.makedirs(cm_save_dir, exist_ok=True)

        train_loader, val_loader, class_names = get_loaders_fold(
            data_dir="./data_cropped/train_eff", 
            fold_idx=fold,
            n_folds=N_FOLDS,
            batch_size=batch_size,      
            img_size=img_size        
        )

        if use_convnext:
            model = timm.create_model(
                'convnext_large',
                pretrained=True,
                num_classes=len(class_names)
            ).to(device)
            # model.set_grad_checkpointing(True) # 很慢，梯度检查点。开了之后一个epoch大概需要20多分钟
        else:
            model = get_model(num_classes=len(class_names), model_name='efficientnet_b3').to(device)

        criterion = FocalLoss(alpha=None, gamma=2.0, label_smoothing=0.1)
        lr = 5e-5 if use_convnext else 1e-4
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5, factor=0.5) # 学习率调度器：监控验证集 Macro-F1，如果不增长就降低学习率

        best_f1 = 0.0
        scaler = GradScaler('cuda')
        for epoch in range(60): # mixup makes to 80 round instead of 60
            model.train()
            running_loss = 0.0
            for inputs, labels in tqdm.tqdm(train_loader, desc=f"Epoch {epoch}"):
                inputs, labels = inputs.to(device), labels.to(device)

                # # --- [新增：应用 Mixup] --- alpha 建议设在 0.2-0.4 之间。如果设为 0，则不启用 mixup
                # inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, alpha=0.2)

                optimizer.zero_grad()

                # outputs = model(inputs)
                # loss = criterion(outputs, labels) # loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam) # loss = criterion(outputs, labels)
                # loss.backward()
                # optimizer.step()

                # 混合精度训练
                with autocast('cuda'):                        # fp16前向，显存减半，主要是为了convnext需要
                    outputs = model(inputs)
                    loss = criterion(outputs, labels) # loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam) # loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                running_loss += loss.item()

            # 验证
            model.eval()
            val_loss = 0.0
            all_preds, all_labels = [], []
            with torch.no_grad():
                for inputs, labels in tqdm.tqdm(val_loader, desc=f"Epoch {epoch} [Val]"):
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    preds = torch.argmax(outputs, 1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            
            # 计算整体 Macro-F1
            avg_val_loss = val_loss / len(val_loader)
            macro_f1 = f1_score(all_labels, all_preds, average='macro')
            
            print(f"\nEpoch {epoch} Summary:")
            print(f"Train Loss: {running_loss/len(train_loader):.4f} | Val Loss: {avg_val_loss:.4f} | Macro-F1: {macro_f1:.4f}")
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Current Learning Rate: {current_lr:.8f}")
            log_message("-" * 30)
            log_message(f"Epoch {epoch} | Train Loss: {running_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Macro-F1: {macro_f1:.4f} | LR: {optimizer.param_groups[0]['lr']:.8f}")
            
            # 🟢 打印每个类别的详细报告 (包括每个类的 F1-score)
            print("\n详细分类报告 (Per-class F1):")
            # classification_report 会输出每个类的 precision, recall, f1-score
            report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
            print(report)
            
            cm = confusion_matrix(all_labels, all_preds)
            plt.figure(figsize=(12, 10))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=class_names, yticklabels=class_names)
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title(f'Epoch {epoch} Confusion Matrix')
            # 保存图片，不要在服务器上 plt.show()
            plt.savefig(f'{cm_save_dir}/confusion_matrix_epoch_{epoch}.png')
            plt.close()

            # 调度器步进
            scheduler.step(macro_f1)
            
            # 保存最佳模型
            if macro_f1 > best_f1:
                best_f1 = macro_f1
                torch.save(model.state_dict(), model_save_path)
                print(f"✅ 保存最佳模型，Macro-F1: {best_f1:.4f}")
                log_message(f"✅ 发现更佳模型! 保存路径: {model_save_path}, Macro-F1: {best_f1:.4f}")


if __name__ == "__main__":
    # 训练EfficientNet-B3：
    # train(use_convnext=False)
    
    # 训练ConvNeXt-Large：
    train(use_convnext=True)