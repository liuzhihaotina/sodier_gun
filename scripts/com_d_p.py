import os
import time
import shutil
import random
import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 尝试导入 DALI
try:
    from nvidia.dali.pipeline import Pipeline
    import nvidia.dali.ops as ops
    import nvidia.dali.types as types
    from nvidia.dali.plugin.pytorch import DALIGenericIterator
    DALI_AVAILABLE = True
except ImportError:
    DALI_AVAILABLE = False
    print("Warning: NVIDIA DALI 库未安装，无法运行 DALI 相关测试。")

# ================= 配置参数 =================
BATCH_SIZE = 128
NUM_WORKERS = 4  # PyTorch DataLoader 线程数
IMAGE_SIZE = 224
NUM_IMAGES = 2000 # 生成测试图片的数量
DATA_DIR = "./dummy_data_benchmark"
EPOCHS = 2

# ================= 1. 数据准备 =================
def create_dummy_data(root_dir, num_images):
    if os.path.exists(root_dir):
        return
    print(f"正在生成 {num_images} 张测试图片到 {root_dir} ...")
    os.makedirs(os.path.join(root_dir, "class_a"), exist_ok=True)
    
    for i in range(num_images):
        # 生成随机 RGB 图片
        img = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        img = Image.fromarray(img)
        img.save(os.path.join(root_dir, "class_a", f"{i}.jpg"))
    print("数据生成完毕。")

# ================= 2. PyTorch DataLoader =================
def get_torch_loader(root_dir, mode='train'):
    if mode == 'train':
        transform = transforms.Compose([
            transforms.RandomResizedCrop(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        shuffle = True
    else: # val/inference
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        shuffle = False

    dataset = datasets.ImageFolder(root_dir, transform=transform)
    # pin_memory=True 对 GPU 训练很重要
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle, 
                        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    return loader

# ================= 3. NVIDIA DALI Pipeline =================
class BenchmarkPipeline(Pipeline):
    def __init__(self, data_dir, batch_size, num_threads, device_id, mode='train'):
        super(BenchmarkPipeline, self).__init__(batch_size, num_threads, device_id, seed=12)
        self.mode = mode
        
        # 读取文件 (CPU)
        self.input = ops.readers.File(file_root=data_dir, random_shuffle=(mode=='train'))
        
        # 解码 (混合设备: CPU/GPU) -> 推荐使用 'mixed' 即使在 GPU 上解码
        self.decode = ops.decoders.Image(device="mixed", output_type=types.RGB)
        
        # 增强操作
        if mode == 'train':
            self.res = ops.RandomResizedCrop(device="gpu", size=IMAGE_SIZE)
            self.cmn = ops.CropMirrorNormalize(device="gpu",
                                               dtype=types.FLOAT,
                                               output_layout=types.NCHW,
                                               mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                               std=[0.229 * 255, 0.224 * 255, 0.225 * 255])
            self.coin = ops.CoinFlip(probability=0.5)
        else:
            self.res = ops.Resize(device="gpu", resize_shorter=256)
            self.cmn = ops.CropMirrorNormalize(device="gpu",
                                               dtype=types.FLOAT,
                                               output_layout=types.NCHW,
                                               mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                               std=[0.229 * 255, 0.224 * 255, 0.225 * 255])

    def define_graph(self):
        jpegs, labels = self.input(name="Reader")
        images = self.decode(jpegs)
        
        if self.mode == 'train':
            images = self.res(images)
            mirror = self.coin()
            images = self.cmn(images, mirror=mirror)
        else:
            images = self.res(images)
            # Center Crop 通常在 CropMirrorNormalize 中通过参数控制，或者单独 Resize 后 Crop
            # 这里简化为直接 Resize+Normalize (模拟推理)
            images = self.cmn(images)
            
        return images, labels

def get_dali_iter(data_dir, mode='train'):
    pipe = BenchmarkPipeline(data_dir, batch_size=BATCH_SIZE, num_threads=NUM_WORKERS, device_id=0, mode=mode)
    pipe.build()
    # DALI 返回的大小可能不完全等于 len(dataset)，drop_last=True 比较公平
    dali_iter = DALIGenericIterator(pipe, ["data", "label"], reader_name="Reader", auto_reset=True)
    return dali_iter, pipe.epoch_size("Reader")

# ================= 4. 测试函数 =================
def benchmark(loader, num_images, name):
    print(f"--- 开始测试: {name} ---")
    
    # 预热 (Warmup)
    print("预热中...")
    start = time.time()
    for _ in loader:
        pass
    
    # 正式计时
    print("正式计时开始...")
    torch.cuda.synchronize()
    start_time = time.time()
    
    count = 0
    for _ in range(EPOCHS):
        for i, data in enumerate(loader):
            # 模拟将数据移动到 GPU (如果是 PyTorch Loader)
            # DALI 出来的已经是 GPU tensor 了
            if isinstance(loader, DataLoader):
                images, labels = data
                images = images.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
            else:
                # DALI iterator 返回的是 list of dict
                batch = data[0]
                images = batch["data"]
                labels = batch["label"]
            
            count += images.shape[0]
            
    torch.cuda.synchronize()
    end_time = time.time()
    
    total_time = end_time - start_time
    throughput = count / total_time
    print(f"[{name}] 总耗时: {total_time:.4f}s, 总图片数: {count}")
    print(f"[{name}] 吞吐量: {throughput:.2f} images/sec")
    print("-" * 30)
    return throughput

# ================= 主程序 =================
if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("错误: 需要 GPU 环境才能进行有意义的对比。")
        exit(1)
        
    create_dummy_data(DATA_DIR, NUM_IMAGES)
    
    # 1. PyTorch 训练模式
    torch_train = get_torch_loader(DATA_DIR, mode='train')
    speed_torch_train = benchmark(torch_train, NUM_IMAGES, "PyTorch Train (CPU preprocess)")
    
    # 2. DALI 训练模式
    if DALI_AVAILABLE:
        dali_train, size = get_dali_iter(DATA_DIR, mode='train')
        speed_dali_train = benchmark(dali_train, size, "DALI Train (GPU preprocess)")
        print(f"训练加速比 (DALI / PyTorch): {speed_dali_train / speed_torch_train:.2f}x")
    
    print("\n" + "="*40 + "\n")

    # 3. PyTorch 推理/评估模式
    torch_eval = get_torch_loader(DATA_DIR, mode='eval')
    speed_torch_eval = benchmark(torch_eval, NUM_IMAGES, "PyTorch Eval (CPU preprocess)")
    
    # 4. DALI 推理/评估模式
    if DALI_AVAILABLE:
        dali_eval, size = get_dali_iter(DATA_DIR, mode='eval')
        speed_dali_eval = benchmark(dali_eval, size, "DALI Eval (GPU preprocess)")
        print(f"推理加速比 (DALI / PyTorch): {speed_dali_eval / speed_torch_eval:.2f}x")
    
    # 清理数据
    # shutil.rmtree(DATA_DIR) 
    print("测试完成。")
