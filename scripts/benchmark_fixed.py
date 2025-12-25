import os
# ==========================================
# 关键修复 1: 禁用 NVML 以解决 nvml error (3)
# 必须在导入 nvidia.dali 之前设置
# ==========================================
os.environ["DALI_DISABLE_NVML"] = "1"

import time
import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 尝试导入 DALI
try:
    import nvidia.dali.fn as fn
    import nvidia.dali.types as types
    from nvidia.dali import pipeline_def
    from nvidia.dali.plugin.pytorch import DALIGenericIterator
    DALI_AVAILABLE = True
except ImportError:
    DALI_AVAILABLE = False
    print("Warning: NVIDIA DALI 库未安装。")

# ================= 配置参数 =================
BATCH_SIZE = 128
NUM_WORKERS = 4
IMAGE_SIZE = 224
NUM_IMAGES = 2000
DATA_DIR = "./dummy_data_benchmark"
EPOCHS = 2

# ================= 1. 数据准备 (自动生成) =================
def create_dummy_data(root_dir, num_images):
    if os.path.exists(root_dir):
        return
    print(f"正在生成 {num_images} 张测试图片到 {root_dir} ...")
    os.makedirs(os.path.join(root_dir, "class_a"), exist_ok=True)
    for i in range(num_images):
        img = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        Image.fromarray(img).save(os.path.join(root_dir, "class_a", f"{i}.jpg"))
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
    else:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        shuffle = False

    dataset = datasets.ImageFolder(root_dir, transform=transform)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle, 
                      num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)

# ================= 3. NVIDIA DALI Pipeline (现代函数式 API) =================
# 关键修复 2: 使用 @pipeline_def 和 fn.* 替代旧的 class Pipeline 和 ops.*
@pipeline_def(batch_size=BATCH_SIZE, num_threads=NUM_WORKERS, device_id=0)
def create_dali_pipeline(data_dir, mode='train'):
    # 读取文件
    jpegs, labels = fn.readers.file(file_root=data_dir, 
                                    random_shuffle=(mode == 'train'), 
                                    name="Reader")
    
    # 解码 (混合设备: CPU读取 -> GPU解码)
    images = fn.decoders.image(jpegs, device="mixed", output_type=types.RGB)
    
    if mode == 'train':
        # 训练增强：随机裁剪 + 翻转 + 归一化
        images = fn.random_resized_crop(images, size=IMAGE_SIZE, device="gpu")
        mirror = fn.random.coin_flip(probability=0.5) # 替代了旧的 ops.CoinFlip
    else:
        # 推理增强：Resize + CenterCrop (通过 crop_mirror_normalize 实现)
        images = fn.resize(images, device="gpu", resize_shorter=256)
        mirror = False
        
    # 归一化 + 格式转换 (HWC -> CHW)
    # 注意：验证集如果需要 CenterCrop，通常在这里设置 crop 参数，或者在 resize 后加 fn.crop
    # 这里为了对齐 PyTorch 的简单逻辑，直接用 CropMirrorNormalize 做最后的处理
    images = fn.crop_mirror_normalize(images, 
                                      device="gpu",
                                      dtype=types.FLOAT,
                                      output_layout=types.NCHW,
                                      crop=(IMAGE_SIZE, IMAGE_SIZE), # 强制裁剪到目标尺寸
                                      mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                      std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
                                      mirror=mirror)
    
    return images, labels

def get_dali_iter(data_dir, mode='train'):
    pipe = create_dali_pipeline(data_dir=data_dir, mode=mode)
    pipe.build()
    # 获取 epoch 大小
    size = pipe.epoch_size("Reader")
    dali_iter = DALIGenericIterator(pipe, ["data", "label"], reader_name="Reader", auto_reset=True)
    return dali_iter, size

# ================= 4. 测试函数 =================
def benchmark(loader, num_images, name):
    print(f"--- 开始测试: {name} ---")
    
    # 预热
    print("预热中...")
    try:
        for _ in loader: break
    except StopIteration: pass
    
    if hasattr(loader, "reset"): loader.reset() # DALI 重置
    
    # 计时
    print("正式计时开始...")
    torch.cuda.synchronize()
    start_time = time.time()
    
    count = 0
    for _ in range(EPOCHS):
        for i, data in enumerate(loader):
            if isinstance(loader, DataLoader):
                images = data[0].cuda(non_blocking=True)
            else:
                # DALI
                images = data[0]["data"]
            count += images.shape[0]
            
    torch.cuda.synchronize()
    end_time = time.time()
    
    throughput = count / (end_time - start_time)
    print(f"[{name}] 吞吐量: {throughput:.2f} images/sec")
    return throughput

# ================= 主程序 =================
if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("错误: 必须有 NVIDIA GPU 才能运行此脚本。")
        exit(1)

    create_dummy_data(DATA_DIR, NUM_IMAGES)
    
    # 1. PyTorch Train
    ts = benchmark(get_torch_loader(DATA_DIR, 'train'), NUM_IMAGES, "PyTorch Train")
    
    # 2. DALI Train
    if DALI_AVAILABLE:
        dl, size = get_dali_iter(DATA_DIR, 'train')
        ds = benchmark(dl, size, "DALI Train")
        print(f"🚀 训练加速比: {ds / ts:.2f}x")

    print("-" * 30)

    # 3. PyTorch Eval
    ts_eval = benchmark(get_torch_loader(DATA_DIR, 'eval'), NUM_IMAGES, "PyTorch Eval")
    
    # 4. DALI Eval
    if DALI_AVAILABLE:
        dl_eval, size = get_dali_iter(DATA_DIR, 'eval')
        ds_eval = benchmark(dl_eval, size, "DALI Eval")
        print(f"🚀 推理加速比: {ds_eval / ts_eval:.2f}x")
