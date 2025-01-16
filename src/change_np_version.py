import datetime
import json
import os
import pickle
import sys
import time

import numpy as np

# from pymycobot.common import ProtocolCode
# from torch import nn
#
# from camera.dslr import WebCam
# from options import Options
# from paint_utils3 import show_img, random_init_painting
# from src.painter import Painter, PERPENDICULAR_QUATERNION
# from my_tensorboard import TensorBoard
# from src.robot import Cobot280, quaternion_to_euler_degrees
def save_ndarray_to_json(dir_path):
    """用本机版本保存到 JSON 文件"""
    names = ['cached_H_coord', "cached_H_canvas" ]
    for name in names:
        H_coord = pickle.load(open(os.path.join(dir_path, f"{name}.pkl"), 'rb'),
                              encoding='latin1')
        print(type(H_coord))
        print(H_coord)
        with open(os.path.join(dir_path, f"{name}.json"), 'w', encoding='utf-8') as f:
            json.dump(H_coord.tolist(), f)
            # 读取 JSON 文件并将其转换为 H_coord numpy ndarray


def reload_ndarray_from_json(dir_path):
    """从 JSON 文件重新加载 numpy ndarray"""
    names = ['cached_H_coord', "cached_H_canvas"]
    for name in names:
        with open(os.path.join(dir_path, f"{name}.json"), 'r', encoding='utf-8') as f:
            H_coord = np.array(json.load(f))  # 将 JSON 数据转换为 numpy ndarray
        print(type(H_coord))
        print(H_coord)
        with open(os.path.join(dir_path, f"{name}.pkl"), 'wb') as f:
            pickle.dump(H_coord, f)


if __name__ == '__main__':
    dir_path = r"D:\code\frida\src\caches\small_brush_uarm_mt"

    # save_ndarray_to_json(dir_path)
    reload_ndarray_from_json(dir_path)
    # exit(0)

