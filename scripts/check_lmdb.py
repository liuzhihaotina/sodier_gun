#!/usr/bin/env python3
"""
检查LMDB数据格式
"""
import lmdb
import struct

def check_lmdb_format(lmdb_path):
    env = lmdb.open(lmdb_path, subdir=False, readonly=True)
    
    with env.begin() as txn:
        # 检查第一个key-value对
        cursor = txn.cursor()
        for key, value in cursor:
            if key == b'__keys__':
                continue
                
            print(f"Key: {key}")
            print(f"Value length: {len(value)} bytes")
            
            # 尝试解析为Caffe2格式 (4字节标签 + 图像数据)
            if len(value) >= 4:
                label = struct.unpack('>I', value[:4])[0]
                print(f"Label (first 4 bytes): {label}")
                print(f"Image data length: {len(value) - 4} bytes")
            break  # 只检查第一个样本

if __name__ == '__main__':
    check_lmdb_format('data/out_train/train.lmdb')