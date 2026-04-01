# train.py
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1" 
import torch
import torch.nn as nn
import torch.optim as optim
from model import get_model
from dataset import get_loaders
from sklearn.metrics import f1_score, classification_report
import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def train():
    # train_loader, val_loader, class_names = get_loaders("./data_cropped/train")
    train_loader, val_loader, class_names = get_loaders(
        data_dir="./data_cropped/train", 
        batch_size=32,      # 如果显存够大可以设 64
        img_size=256        # 必须和你 dataset.py 里的 Resize 一致
    )
    model = get_model(num_classes=len(class_names), model_name='resnet50').to(device)
    
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1) # 传入权重
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2, factor=0.5) # 学习率调度器：监控验证集 Macro-F1，如果不增长就降低学习率

    best_f1 = 0.0
    for epoch in range(50): # 20 完全不够-> 50 epochs，给模型更多时间学习
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm.tqdm(train_loader, desc=f"Epoch {epoch}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
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
        plt.savefig(f'./CNN_models/confusion_matrix/confusion_matrix_epoch_{epoch}.png')
        plt.close()

        # 调度器步进
        scheduler.step(macro_f1)
        
        # 保存最佳模型
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            torch.save(model.state_dict(), "best_wbc_model.pth")
if __name__ == "__main__":
    train()