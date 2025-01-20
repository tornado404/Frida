import datetime
import json
import os
import pickle
import sys
import time

import numpy as np

from pymycobot.common import ProtocolCode
from torch import nn

from camera.dslr import WebCam
from options import Options
from paint_utils3 import show_img, random_init_painting, canvas_to_global_coordinates
import sys
sys.path.append(r"D:\code\frida")
sys.path.append(r"D:\code\frida\src")
from src.painter import Painter, PERPENDICULAR_QUATERNION
from my_tensorboard import TensorBoard
from src.robot import Cobot280, quaternion_to_euler_degrees
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

# 在工程根目录执行
# python.exe src/test_camera.py --robot mycobot280pi --materials_json materials_ink.json --use_cache --cache_dir src/caches/small_brush_cobot --render_height 256 --dont_retrain_stroke_model --num_strokes 100 --n_colors 2 --objective clip_conv_loss --objective_data src/inputs/1.png --objective_weight 1.0 --init_optim_iter 1500 --lr_multiplier 2.0
if __name__ == '__main__':
    dir_path = r"D:\code\frida\src\caches\cobot280"

    opt = Options()
    opt.gather_options()

    date_and_time = datetime.datetime.now()
    run_name = '' + date_and_time.strftime("%m_%d__%H_%M_%S")
    opt.writer = TensorBoard('{}/{}'.format(opt.tensorboard_dir, run_name))
    opt.writer.add_text('args', str(sys.argv), 0)

    painter = Painter(opt)
    opt = painter.opt # This necessary?
    painter.to_neutral()
    print('\n画笔尖校准\n')
    print('画笔应该位于画布的正中心。')
    print('使用 "w" 和 "s" 键将画笔设置为刚好接触画布。')

    p = canvas_to_global_coordinates(0.5, 0.5, painter.opt.INIT_TABLE_Z, painter.opt)
    painter.set_height(p[0], p[1], painter.opt.INIT_TABLE_Z, 0.01)[2]

    # angle = quaternion_to_euler_degrees(PERPENDICULAR_QUATERNION)
    # print("PERPENDICULAR_QUATERNION angle is ", angle)
 
    # from pymycobot import MyCobotSocket
    # mc = MyCobotSocket("192.168.31.7", 9000)
    # # 贴近纸面中心 [278.1, -63.8, 93.1, 180, 0, -45]
    # pos = mc.get_coords()
    # print(pos)
    # time.sleep(5)
    # # [50.3, -63.5, 410.2, -90.86, -44.54, -88.16]
    # mc.sync_send_angles([0, 45, 0, 0, 0, -45], 20)
    # time.sleep(5)
    # pos = mc.get_coords()
    # print("reset: ", pos)
    # # mc._mesg(0x1A,1)
    # mc.sync_send_coords([278.1, -63.8, 93.1, 180, 0, -45], 10, 0)
    # time.sleep(5)
    # pos = mc.get_position()
    # print(pos)
    #


    
    # mc.move_to_position(0, 0, 0)

    # opt = Options()
    # opt.gather_options()
    #
    # date_and_time = datetime.datetime.now()
    # run_name = '' + date_and_time.strftime("%m_%d__%H_%M_%S")
    # opt.writer = TensorBoard('{}/{}'.format(opt.tensorboard_dir, run_name))
    # opt.writer.add_text('args', str(sys.argv), 0)
    #
    # painter = Painter(opt)
    # opt = painter.opt # This necessary?


    # opt = Options()
    # opt.gather_options()
    # wc = WebCam(opt)
    # wc.debug = True
    # wc.calibrate_canvas(use_cache=opt.use_cache)
    # img = wc.get_canvas()
    # show_img(img/255.,
    #              title="初步计划完成。准备开始绘画。 \
    #                 确保提供混合颜料，然后退出以 \
    #                 开始绘画。")
    #
    
