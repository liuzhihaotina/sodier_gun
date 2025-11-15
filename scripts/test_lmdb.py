import lmdb
import json
import struct
from PIL import Image
import io

def inspect_lmdb(lmdb_path):
    env = lmdb.open(lmdb_path, subdir=False, readonly=True, lock=False, readahead=False)
    
    with env.begin() as txn:
        # 读取元数据
        keys = json.loads(txn.get(b'__keys__').decode('utf-8'))
        class_map = json.loads(txn.get(b'__class_map__').decode('utf-8'))
        
        print("Class map:", class_map)
        print("Total keys:", len(keys))
        print("First 10 keys:", keys[:10])
        
        # 读取第一个样本
        first_key = keys[0].encode('ascii')
        value = txn.get(first_key)
        
        # 解析标签和图像数据
        label = struct.unpack('>I', value[:4])[0]  # 前4字节是标签
        img_data = value[4:]  # 剩余的是图像数据
        
        print(f"First sample - Key: {keys[0]}, Label: {label}")
        print(f"Image data size: {len(img_data)} bytes")
        
        # 显示图像（可选）
        img = Image.open(io.BytesIO(img_data))
        print(f"Image size: {img.size}, Mode: {img.mode}")
        # img.show()  # 显示图像
        
    env.close()

# 使用示例
inspect_lmdb('data/out_train/train.lmdb')