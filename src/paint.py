#! /usr/bin/env python
##########################################################
#################### Copyright 2022 ######################
################ by Peter Schaldenbrand ##################
### The Robotics Institute, Carnegie Mellon University ###
################ All rights reserved. ####################
##########################################################
import os.path
import sys
import time

import cv2
import datetime
import numpy as np

import torch
from torch import nn

from paint_utils3 import canvas_to_global_coordinates, get_colors, nearest_color, random_init_painting, save_colors, show_img

from painter import Painter
from options import Options
# from paint_planner import paint_planner_new

from my_tensorboard import TensorBoard
from painting_optimization import load_objectives_data, optimize_painting

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#mengtign  --robot mycobot280pi --use_cache --cache_dir caches/sss_test --ink --render_height 256 --dont_retrain_stroke_model --num_strokes 100 --objective clip_conv_loss --objective_data inputs/4.jpg --objective_weight 1.0 --init_optim_iter 400 --lr_multiplier 2.0
#mengtign  --use_cache --cache_dir caches/sss_test --simulate --ink --render_height 256 --dont_retrain_stroke_model --num_strokes 100 --objective clip_conv_loss --objective_data inputs/4.jpg --objective_weight 1.0 --init_optim_iter 2000 --lr_multiplier 2.0
# --simulate --render_height 256 --use_cache --cache_dir caches/sharpie_short_strokes --dont_retrain_stroke_model --objective clip_conv_loss --objective_data caches/1.png --robot mycobot280pi --objective_weight 1.0 --lr_multiplier 0.4 --num_strokes 100 --optim_iter 400 --n_colors 1
# --render_height 256 --use_cache --cache_dir caches/small_brush_test --dont_retrain_stroke_model --objective clip_conv_loss --objective_data caches/1.png --robot mycobot280pi --objective_weight 1.0 --lr_multiplier 0.4 --num_strokes 100 --optim_iter 800 --n_colors 1

# D:\ProgramData\miniconda3\envs\frida\python.exe D:\code\frida\src\paint.py --robot mycobot280pi --materials_json materials_ink.json  --use_cache --cache_dir src/caches/cobot280 --ink --render_height 256 --dont_retrain_stroke_model --num_strokes 100 --objective clip_conv_loss --objective_data src/inputs/4.jpg --objective_weight 1.0 --init_optim_iter 400 --lr_multiplier 2.0
if __name__ == '__main__':
    opt = Options()
    opt.gather_options()

    date_and_time = datetime.datetime.now()
    run_name = '' + date_and_time.strftime("%m_%d__%H_%M_%S")
    opt.writer = TensorBoard('{}/{}'.format(opt.tensorboard_dir, run_name))
    opt.writer.add_text('args', str(sys.argv), 0)

    painter = Painter(opt)
    opt = painter.opt # This necessary?

    # for i in range(2):
    #     print('获取第', i, '种颜色')
    #     painter.get_paint(i)
    # painter.rub_brush_on_rag()
    #
    #
    # exit(0)
    # painter.to_neutral()
    if not opt.simulate:
        try:
            painter.dip_brush_in_water()
            input('Make sure blank canvas is exposed. Press enter when you are ready for the paint planning to start. Use tensorboard to see which colors to paint.')
        except SyntaxError:
            pass

    # paint_planner_new(painter)

    # painter.to_neutral()

    w_render = int(opt.render_height * (opt.CANVAS_WIDTH_M/opt.CANVAS_HEIGHT_M))
    h_render = int(opt.render_height)
    opt.w_render, opt.h_render = w_render, h_render

    consecutive_paints = 0
    consecutive_strokes_no_clean = 0
    curr_color = -1

    color_palette = None
    if opt.use_colors_from is not None:
        color_palette = get_colors(cv2.resize(cv2.imread(opt.use_colors_from)[:,:,::-1], (256, 256)), n_colors=opt.n_colors).to(device)
        opt.writer.add_image('paint_colors/using_colors_from_input', save_colors(color_palette), 0)
    dark_black = [0, 0, 0]  # 浓黑色
    light_ink = [50, 50, 50]  # 淡墨色
    color_palette = torch.tensor([dark_black, light_ink], dtype=torch.float32).to(device)

    print("current_canvas = painter.camera.get_canvas_tensor(h=h_render,w=w_render).to(device) / 255.")
    current_canvas = painter.camera.get_canvas_tensor(h=h_render,w=w_render).to(device) / 255.

    load_objectives_data(opt)

    painting = random_init_painting(opt, current_canvas, opt.num_strokes, ink=opt.ink)
    painting.to(device)

    # Do the initial optimization
    if not os.path.exists('painting_data.json'):
        painting, color_palette = optimize_painting(opt, painting,
                    optim_iter=opt.init_optim_iter, color_palette=color_palette)

        import json

        with open('painting_data.json', 'w', encoding='utf-8') as f:
            json.dump({
                'painting': {
                    'brush_strokes': [painting.pop().to_dict() for _ in range(len(painting.brush_strokes))],
                    # 使用 pop 遍历并序列化 brush_strokes
                },
            }, f, ensure_ascii=False)


    # 从文件恢复 painting
    from src.brush_stroke import BrushStroke
    with open('painting_data.json', 'r', encoding='utf-8') as f:  # 读取文件
        import json
        data = json.load(f)
        painting = random_init_painting(opt, current_canvas, opt.num_strokes, ink=opt.ink)
        strokes_array = []
        for stroke_data in data['painting']['brush_strokes']:  # 遍历反序列化 brush_strokes
            strokes_array.append(BrushStroke.from_dict(stroke_data, opt))
        painting.brush_strokes = nn.ModuleList(strokes_array)
        if 'color_palette' in data['painting']:
            color_palette_int = [[int(c * 255) for c in color] for color in data['painting']['color_palette']]
            print("color_palette_int", color_palette_int)
            color_palette = torch.tensor(color_palette_int, dtype=torch.float32).to(device)


    if not painter.opt.simulate:
        show_img(current_canvas, "初始画布效果")
        show_img(painting.background_img, "painting.background_img")
        show_img(painter.camera.get_canvas()/255.,
                 title="Initial plan complete. Ready to start painting. \
                    Ensure mixed paint is provided and then exit this to \
                    start painting.")


    strokes_per_adaptation = int(len(painting) / opt.num_adaptations)
    # for adaptation_it in range(opt.num_adaptations):


    if len(painting) == 0:
        print("No strokes to paint. Exiting.")
        exit(0)
    strokes_array = []
    for stroke_ind in range(min(len(painting), strokes_per_adaptation)):
        stroke = painting.pop()
        strokes_array.append(stroke)

    temp_strokes = strokes_array.copy()
    # 记录每个笔画的颜色索引
    color_index_list = []
    for stroke_ind, stroke in enumerate(temp_strokes):
        color_ind, _ = nearest_color(stroke.color_transform.detach().cpu().numpy(),
                                 color_palette.detach().cpu().numpy())
        color_index_list.append(color_ind)
    # copy array and walk through it by different color
    print("color_index_list", color_index_list)
    for index, color in enumerate(color_palette):
        # 遍历 color_palette 中的每种颜色, 每一轮只画一种颜色，画完之后执行换颜色操作
        temp_strokes = strokes_array.copy()
        consecutive_paints = 0
        for stroke_ind, stroke in enumerate(temp_strokes):
            # Clean paint brush and/or get more paint
            if color_index_list[stroke_ind] != index:
                # 如果颜色不匹配，则跳过
                continue
            if not painter.opt.ink:
                new_paint_color = False
                # if consecutive_strokes_no_clean > 12:
                #     painter.clean_paint_brush()
                #     painter.clean_paint_brush()
                #     consecutive_strokes_no_clean = 0
                #     curr_color = color_ind
                #     new_paint_color = True
                if consecutive_paints % opt.how_often_to_get_paint == 0 or new_paint_color:
                    painter.get_paint(index)
                    painter.rub_brush_on_rag()
                    consecutive_paints = 0

            # Convert the canvas proportion coordinates to meters from robot
            x, y = stroke.transformation.xt.item() * 0.5 + 0.5, stroke.transformation.yt.item() * 0.5 + 0.5
            y = 1 - y
            x, y = min(max(x, 0.), 1.), min(max(y, 0.), 1.)  # safety
            x_glob, y_glob, _ = canvas_to_global_coordinates(x, y, None, painter.opt)

            # Runnit
            stroke.execute(painter, x_glob, y_glob, stroke.transformation.a.item())
            consecutive_paints += 1

        painter.clean_paint_brush()
        painter.clean_paint_brush()
        consecutive_strokes_no_clean = 0

        # 如果不是最后一种颜色，则获取下一种颜色
        if index != len(color_palette) - 1:
            painter.get_paint(index + 1)
        painter.rub_brush_on_rag()
        consecutive_paints = 0

    #######################
    ### Update the plan ###
    #######################
    painter.dip_brush_in_water()
    time.sleep(15)
    current_canvas = painter.camera.get_canvas_tensor(h=h_render, w=w_render).to(device) / 255.
    painting.background_img = current_canvas
    painting, _ = optimize_painting(opt, painting,
                                    optim_iter=opt.optim_iter, color_palette=color_palette)

    # to_video(plans, fn=os.path.join(opt.plan_gif_dir,'sim_canvases{}.mp4'.format(str(time.time()))))
    # with torch.no_grad():
    #     save_image(painting(h*4,w*4, use_alpha=False), os.path.join(opt.plan_gif_dir, 'init_painting_plan{}.png'.format(str(time.time()))))

    painter.robot.good_night_robot()


