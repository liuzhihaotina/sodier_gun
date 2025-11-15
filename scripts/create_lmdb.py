#!/usr/bin/env python3
"""
将按类目录组织的图片数据集写入 LMDB。
存储格式（value）:
  4-byte big-endian unsigned int label + image bytes (原始 encoded bytes, e.g. JPEG/PNG)

LMDB 会包含以下特殊 key:
  b'__keys__' -> JSON list of keys (str)
  b'__class_map__' -> JSON dict class_name -> label_id
"""
import os
import sys
import lmdb
import argparse
import json
import struct
from pathlib import Path
from tqdm import tqdm

def gather_images(src_dir):
    src = Path(src_dir)
    class_dirs = [d for d in sorted(src.iterdir()) if d.is_dir()]
    class_map = {d.name: i for i, d in enumerate(class_dirs)}
    entries = []
    for class_dir in class_dirs:
        label = class_map[class_dir.name]
        for p in sorted(class_dir.rglob('*')):
            if p.is_file():
                # simple check for common image extensions
                if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp', '.webp'):
                    entries.append((str(p), label))
    return entries, class_map

def estimate_map_size(entries):
    total = 0
    for p, _ in entries:
        total += os.path.getsize(p)
    # give some headroom
    return int(total * 4) + 1024 * 1024

def write_lmdb(entries, class_map, out_path):
    map_size = estimate_map_size(entries)
    env = lmdb.open(out_path, map_size=map_size, subdir=False, lock=True, readahead=True)
    with env.begin(write=True) as txn:
        keys = []
        for idx, (p, label) in enumerate(tqdm(entries, desc="Writing images to LMDB")):
            with open(p, 'rb') as f:
                img = f.read()
            key = '{:08}'.format(idx).encode('ascii')
            # pack label as 4-byte big-endian unsigned int
            value = struct.pack('>I', label) + img
            txn.put(key, value)
            keys.append(key.decode('ascii'))
        # store keys and class_map as metadata
        txn.put(b'__keys__', json.dumps(keys).encode('utf-8'))
        txn.put(b'__class_map__', json.dumps(class_map).encode('utf-8'))
    env.sync()
    env.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', required=True, help='dataset root dir (class subdirs)')
    parser.add_argument('--out', required=True, help='output lmdb path (file)')
    args = parser.parse_args()
    entries, class_map = gather_images(args.src)
    print("Found {} images, {} classes".format(len(entries), len(class_map)))
    write_lmdb(entries, class_map, args.out)
    print("Wrote LMDB to", args.out)
    # also write class_map file beside lmdb for convenience
    base = args.out
    with open(base + '.classes.json', 'w') as f:
        json.dump(class_map, f, indent=2)
    print("Wrote class map to", base + '.classes.json')

if __name__ == '__main__':
    # 临时设置参数
    sys.argv = ['scripts/create_lmdb.py', '--src', 'data/train', '--out', 'data/out_train/train.lmdb']
    main()