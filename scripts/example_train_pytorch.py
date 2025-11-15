# from dali_lmdb_pipeline import DALILMDBTrainPipeline, DALILMDBValPipeline
#!/usr/bin/env python3
"""
CPU版本训练脚本
"""
import torch
import torch.nn as nn
import torch.optim as optim
import sys

# 先测试管道
try:
    from dali_lmdb_pipeline import test_cpu_pipeline
    if test_cpu_pipeline():
        from dali_lmdb_pipeline import CPUPipeline
        print("Using CPU decoding + GPU processing")
    else:
        print("Falling back to CPU-only processing")
        # from dali_lmdb_pipeline_only import CPUOnlyPipeline
except:
    print("Using CPU-only processing as fallback")
    # from dali_lmdb_pipeline_only import CPUOnlyPipeline


class TinyModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, num_classes)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def main():
    print("Starting training with CPU pipeline...")
    
    # 创建管道
    try:
        train_pipe = CPUPipeline(
            lmdb_path='data/out_train/train.lmdb',
            batch_size=8,
            device_id=0
        )
    except:
        print('train_pipe = CPUPipeline(...) failed, using CPUOnlyPipeline instead.')
        # train_pipe = CPUOnlyPipeline(
        #     lmdb_path='data/out_train/train.lmdb',
        #     batch_size=8
        # )
    
    train_pipe.build()
    train_loader = train_pipe.get_iter()
    
    # 创建模型
    device = torch.device('cuda:0')
    model = TinyModel(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    
    # 训练循环
    model.train()
    for epoch in range(2):
        total_loss = 0
        batch_count = 0
        
        for i, data in enumerate(train_loader):
            # if i >= 10:  # 只训练10个batch进行测试
            #     break
                
            batch = data[0]
            images = batch['images'].to(device)
            labels = batch['labels'].squeeze().long().to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
            
            if i % 2 == 0:
                print(f'Epoch {epoch}, Batch {i}, Loss: {loss.item():.4f}')
        
        if batch_count > 0:
            avg_loss = total_loss / batch_count
            print(f'Epoch {epoch} finished, Avg Loss: {avg_loss:.4f}')
    
    print("Training completed!")


if __name__ == '__main__':
    main()