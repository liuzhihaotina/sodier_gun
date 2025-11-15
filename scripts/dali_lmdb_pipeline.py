#!/usr/bin/env python3
"""
LMDB + DALI (纯CPU解码版本，完全避免NVML错误)
"""
import lmdb
import json
import struct
import random
import numpy as np
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.pipeline import pipeline_def
from nvidia.dali.plugin.pytorch import DALIGenericIterator


class SimpleLMDBReader:
    def __init__(self, lmdb_path, batch_size):
        self.env = lmdb.open(lmdb_path, subdir=False, readonly=True)
        with self.env.begin() as txn:
            self.keys = [k.decode() for k, _ in txn.cursor() if not k.startswith(b'__')]
        self.batch_size = batch_size
        self.idx = 0
        random.shuffle(self.keys)
        print(f"Loaded {len(self.keys)} samples from LMDB")
    
    def __iter__(self):
        self.idx = 0
        random.shuffle(self.keys)
        return self
    
    def __next__(self):
        if self.idx >= len(self.keys):
            raise StopIteration
        
        batch_images = []
        batch_labels = []
        
        for _ in range(self.batch_size):
            if self.idx >= len(self.keys):
                break
                
            key = self.keys[self.idx]
            with self.env.begin() as txn:
                raw = txn.get(key.encode())
            
            if raw is None:
                self.idx += 1
                continue
                
            label = struct.unpack('>I', raw[:4])[0]
            img_data = raw[4:]
            
            batch_images.append(np.frombuffer(img_data, dtype=np.uint8))
            batch_labels.append(np.array(label, dtype=np.int64))
            self.idx += 1
        
        if not batch_images:
            raise StopIteration
            
        return batch_images, batch_labels


# 全局reader实例
_reader = None

def data_source():
    global _reader
    try:
        return next(_reader)
    except StopIteration:
        _reader = iter(_reader)
        return next(_reader)


@pipeline_def
def cpu_pipeline(lmdb_path, bs, img_size=224):
    global _reader
    _reader = SimpleLMDBReader(lmdb_path, bs)
    
    # 使用CPU解码
    jpegs, labels = fn.external_source(
        source=data_source,
        num_outputs=2,
        batch=True,
        parallel=False
    )
    
    # 关键修改：使用CPU解码，然后传输到GPU
    images = fn.decoders.image(jpegs, device="cpu", output_type=types.RGB)
    
    # 将图像传输到GPU进行后续处理
    images = images.gpu()
    images = fn.resize(images, resize_shorter=img_size, device="gpu")
    images = fn.crop_mirror_normalize(
        images,
        device="gpu",
        dtype=types.FLOAT,
        output_layout=types.NCHW,
        mean=[0.485*255, 0.456*255, 0.406*255],
        std=[0.229*255, 0.224*255, 0.225*255],
        crop=(img_size, img_size)
    )
    
    return images, labels


class CPUPipeline:
    def __init__(self, lmdb_path, batch_size, device_id=0):
        self.pipe = cpu_pipeline(
            lmdb_path=lmdb_path,
            bs=batch_size,
            device_id=device_id,
            num_threads=4,
            batch_size=batch_size
        )
    
    def build(self):
        self.pipe.build()
    
    def get_iter(self):
        return DALIGenericIterator(
            pipelines=[self.pipe],
            output_map=['images', 'labels'],
            auto_reset=True
        )


def test_cpu_pipeline():
    try:
        print("Testing CPU pipeline...")
        pipe = CPUPipeline('data/out_train/train.lmdb', 4, 0)
        pipe.build()
        
        loader = pipe.get_iter()
        data = next(loader)
        batch = data[0]
        
        print(f"✓ Success! Images: {batch['images'].shape}, Labels: {batch['labels'].shape}")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False