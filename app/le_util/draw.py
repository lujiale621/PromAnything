import base64
import time

import cv2
import numpy as np
import pynvml

from matplotlib import pyplot as plt

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))

        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack((img, m*0.35)))
def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))
    ax.text(x0, y0, label)
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
def base64_to_cv2(base64_string):
    # 将 base64 字符串解码为字节数组
    img_data = base64.b64decode(base64_string)
    # 从字节数组中读取图像数据
    nparr = np.frombuffer(img_data, np.uint8)
    # 将图像数据解码为 OpenCV 数据类型
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img
def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Time elapsed: {(end_time - start_time) * 1000:.3f} milliseconds")
        return result
    return wrapper
def GPUINFO():
    # 初始化NVML
    pynvml.nvmlInit()

    # 获取GPU数量
    deviceCount = pynvml.nvmlDeviceGetCount()

    # 遍历所有GPU
    for i in range(deviceCount):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)

        # 获取显卡型号
        name = pynvml.nvmlDeviceGetName(handle)
        print(f"GPU {i}: {name}")

        # 获取当前GPU使用情况
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        print(f"GPU {i} Memory Usage:")
        print(f"    Total: {meminfo.total / 1024 / 1024} MB")
        print(f"    Used: {meminfo.used / 1024 / 1024} MB")
        print(f"    Free: {meminfo.free / 1024 / 1024} MB")
        print(f"    Percent: {meminfo.used / meminfo.total * 100}%")

    # 清理NVML
    pynvml.nvmlShutdown()