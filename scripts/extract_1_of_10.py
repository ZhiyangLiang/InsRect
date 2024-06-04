# 以下为该数据集如何从ImageNet原始数据中抽取的脚本

import shutil, os
import random

src_root = "imagenet/train/"           # 原始的ImageNet解压后文件夹，下面有999个文件夹（即分类），每个文件夹中为图片
dst_root = "imagenet/1_of_10_train/"   # 从原始ImageNet每个文件夹（即分类）中抽取1/10的图片数据放在1_of_10_train文件夹下
list_file = "imagenet/train.txt"      # 原始ImageNet的所有图片文件以及类别的列表
'''
# cat imagenet-data/train.txt
n01440764/n01440764_10026.JPEG 0
n01440764/n01440764_10027.JPEG 0
n01440764/n01440764_10029.JPEG 0
n01440764/n01440764_10040.JPEG 0
n01440764/n01440764_10042.JPEG 0
n01440764/n01440764_10043.JPEG 0
n01440764/n01440764_10048.JPEG 0
n01440764/n01440764_10066.JPEG 0
n01440764/n01440764_10074.JPEG 0
n01440764/n01440764_1009.JPEG 0
......
'''

lines = {}
with open(list_file) as fi:
    for l in fi:
        path, cid = l.strip().split(" ")
        cid = int(cid)
        if cid not in lines:
            lines[cid] = []
        lines[cid].append(path)

for i in range(1000):
    paths = lines[i]
    origin_num = len(paths)
    num = int(origin_num / 10)
    paths = random.choices(paths, k=num)
    print("%d / %d" % (num, origin_num))
    for p in paths:
        src = os.path.join(src_root, p)
        dst = os.path.join(dst_root, p)
        shutil.copy(src, dst)

