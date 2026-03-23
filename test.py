import os
import torch
import pandas as pd
import numpy as np
import pydicom
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F

# ==========================================
# 1. 数据集类定义 (数据集处理)
# ==========================================
class RSNADataset(Dataset):
    """
    RSNA 肺炎检测数据集类
    """
    def __init__(self, csv_file, img_dir, transforms=None):
        self.img_dir = img_dir
        self.transforms = transforms
        # 读取标注文件
        self.annotations = pd.read_csv(csv_file)
        # 按 patientId 聚合边界框 
        self.image_ids = self.annotations['patientId'].unique()

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_path = os.path.join(self.img_dir, f"{img_id}.dcm")
        
        # 使用 pydicom 读取医学影像 
        ds = pydicom.dcmread(img_path)
        pixel_array = ds.pixel_array
        
        # 归一化并转为 3 通道 (Faster R-CNN 预训练模型要求)
        img = pixel_array.astype(np.float32) / 255.0
        img = np.stack([img] * 3, axis=0) # [1, H, W] -> [3, H, W]
        img_tensor = torch.tensor(img)

        # 获取该图像所有的边界框
        records = self.annotations[self.annotations['patientId'] == img_id]
        boxes = []
        labels = []
        
        for _, row in records.iterrows():
            if row['Target'] == 1:  # 只有存在肺部浑浊才添加框
                x, y, w, h = row['x'], row['y'], row['width'], row['height']
                # 将 [x, y, w, h] 转换为 [x1, y1, x2, y2]
                boxes.append([x, y, x + w, y + h])
                labels.append(1) # 类别1: Lung Opacity

        # 处理无病灶情况 (Target=0)
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx])
        }

        return img_tensor, target

    def __len__(self):
        return len(self.image_ids)

# ==========================================
# 2. 模型导入 
# ==========================================
def get_model(num_classes):
    # 加载预训练的 Faster R-CNN (ResNet-50-FPN)
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    
    # 修改分类头以匹配我们的类别数 (背景 + 肺炎)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model

# ==========================================
# 3. 训练演示流程
# ==========================================
def main():
    # 参数设置
    num_classes = 2 # 0: 背景, 1: 肺部浑浊
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # 实例化模型
    model = get_model(num_classes)
    model.to(device)

    # 模拟数据整理 (在实际使用中替换为真实路径)
    # dataset = RSNADataset(csv_file='train_labels.csv', img_dir='stage_2_train_images/')
    
    # 模拟一个训练批次的数据 (Collate Function 说明)
    def collate_fn(batch):
        return tuple(zip(*batch))

    # 演示：如何进行一次前向传播
    model.train() # 训练模式下返回损失
    
    # 构造模拟输入 (Tensor 列表)
    images = [torch.rand(3, 1024, 1024).to(device), torch.rand(3, 1024, 1024).to(device)]
    targets = [
        {
            'boxes': torch.tensor([[100, 100, 200, 200]], dtype=torch.float32).to(device),
            'labels': torch.tensor([1], dtype=torch.int64).to(device)
        },
        {
            'boxes': torch.tensor([[150, 150, 300, 400]], dtype=torch.float32).to(device),
            'labels': torch.tensor([1], dtype=torch.int64).to(device)
        }
    ]

    # 计算损失
    loss_dict = model(images, targets)
    losses = sum(loss for loss in loss_dict.values())
    
    print(f"损失组成: {loss_dict}")
    print(f"总损失: {losses.item():.4f}")

    # 演示：如何进行预测
    model.eval()
    with torch.no_grad():
        prediction = model([images[0]])
        print("\n预测结果示例:")
        print(f"预测框数量: {len(prediction[0]['boxes'])}")
        print(f"置信度分数: {prediction[0]['scores'][:5]}") # 前5个最高分

if __name__ == "__main__":
    main()
