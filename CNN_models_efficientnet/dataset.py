# dataset.py
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
import numpy as np

# def get_loaders(data_dir, batch_size=64):
#     # CNN 实时增强策略
#     train_transform = transforms.Compose([
#         transforms.Resize((256, 256)),
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomVerticalFlip(),
#         transforms.RandomRotation(20),
#         # 模拟染色差异的关键：ColorJitter
#         transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ])

#     val_transform = transforms.Compose([
#         transforms.Resize((256, 256)),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ])

#     full_dataset = datasets.ImageFolder(data_dir, transform=train_transform)
#     print("Class to Index mapping:", full_dataset.class_to_idx)
    
#     # 划分训练/验证
#     train_size = int(0.8 * len(full_dataset))
#     val_size = len(full_dataset) - train_size
#     train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
#     val_dataset.dataset.transform = val_transform

#     # 处理类别不平衡：使用 WeightedRandomSampler 替代物理过采样
#     targets = [full_dataset.targets[i] for i in train_dataset.indices]
#     class_sample_count = np.array([len(np.where(targets == t)[0]) for t in np.unique(targets)])
#     weight = 1. / class_sample_count
#     samples_weight = torch.from_numpy(np.array([weight[t] for t in targets])).double()
#     sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

#     train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=4)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
#     return train_loader, val_loader, full_dataset.classes



# 顺序必须严格对应：['BA', 'BL', 'BNE', 'EO', 'LY', 'MMY', 'MO', 'MY', 'PC', 'PLY', 'PMY', 'SNE', 'VLY']
# counts = [415, 2012, 391, 861, 8101, 360, 2746, 441, 68, 11, 114, 13015, 366] 
# 少数类索引：BNE(2), MMY(5), PC(8), PLY(9), PMY(10), VLY(12)
# 这些类在混淆矩阵中表现较差或样本极少
MINORITY_CLASSES = [2, 5, 8, 9, 10, 12]

class MinorityAugDataset(Dataset):
    """
    包装类：根据类别标签动态选择增强强度
    """
    def __init__(self, subset, base_transform, strong_transform):
        self.subset = subset
        self.base_transform = base_transform
        self.strong_transform = strong_transform
        
    def __getitem__(self, index):
        # 从 Subset 中获取原始 PIL 图和标签
        # 注意：ImageFolder 必须在外部初始化时 transform=None
        img, label = self.subset[index]
        
        if label in MINORITY_CLASSES:
            return self.strong_transform(img), label
        else:
            return self.base_transform(img), label
            
    def __len__(self):
        return len(self.subset)

def get_loaders(data_dir, batch_size=64, img_size=256):
    # --- A. 基础增强 (大类使用) ---
    base_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # --- B. 剧烈增强 (少数类使用) ---
    strong_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(180), # 180度大幅度旋转
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2)), # 随机缩放平移
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)), # 模糊处理，对抗过拟合
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), # 更强的变色
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # --- C. 验证集转换 ---
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 加载物理文件夹 (不设 transform)
    full_dataset = datasets.ImageFolder(data_dir, transform=None)
    
    # 划分索引
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_indices, val_indices = torch.utils.data.random_split(
        range(len(full_dataset)), [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # 构建带差异化增强的训练集
    train_subset = torch.utils.data.Subset(full_dataset, train_indices)
    train_dataset = MinorityAugDataset(train_subset, base_transform, strong_transform)
    
    # 构建验证集
    val_subset = torch.utils.data.Subset(full_dataset, val_indices)
    val_dataset = MinorityAugDataset(val_subset, val_transform, val_transform)

    # 计算采样权重 (保持你已有的高效 Sampler)
    targets = [full_dataset.targets[i] for i in train_indices]
    class_sample_count = np.array([len(np.where(targets == t)[0]) for t in np.unique(targets)])
    weight = 1. / class_sample_count
    samples_weight = torch.from_numpy(np.array([weight[t] for t in targets])).double()
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader, full_dataset.classes
