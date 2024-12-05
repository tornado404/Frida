#! /usr/bin/env python
##########################################################
#################### 版权 2023 ##########################
################ 彼得·沙尔登布兰德 #####################
### 卡内基梅隆大学机器人研究所 #######################
################## 保留所有权利 #########################
##########################################################

import datetime
import random
import sys
import time
import cv2
from matplotlib import pyplot as plt
import numpy as np
import torch
from torchvision.transforms import Resize
from tqdm import tqdm
from PIL import Image
import os

# 设置 Hugging Face 的缓存路径
os.environ['HF_DATASETS_CACHE'] = 'G:/huggingface/datasets'
os.environ['HF_HOME'] = 'G:/huggingface'
os.environ['HF_HUB_OFFLINE'] = 'False'
from cofrida import get_instruct_pix2pix_model
from paint_utils3 import canvas_to_global_coordinates, format_img, get_colors, initialize_painting, nearest_color, \
    random_init_painting, save_colors, show_img
from painting_optimization import optimize_painting

from painter import Painter
from options import Options
from my_tensorboard import TensorBoard

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
if not torch.cuda.is_available():
    print('使用 CPU..... 祝你好运')


def get_cofrida_image_to_draw(cofrida_model, curr_canvas_pil, n_ai_options):
    ''' 创建一个 GUI 以允许用户查看不同的 CoFRIDA 选项并选择 '''
    satisfied = False
    while (not satisfied):
        text_prompt = None
        try:
            text_prompt = """masterpiece, (best quality), highly detailed, ultra-detailed, white background, (minimal black), a Chinese mountain top, (ink painting style:1.2), misty atmosphere, (traditional brushstroke texture), serene, (high contrast), (flowing lines), subtle shading, intricate details, vast emptiness, (classic Chinese aesthetic), balance of light and shadow."""
        except SyntaxError:
            continue  # 没有输入

        with torch.no_grad():
            target_imgs = []
            for op in tqdm(range(n_ai_options), desc="CoFRIDA 生成选项"):
                target_imgs.append(cofrida_model(
                    text_prompt,
                    curr_canvas_pil,
                    num_inference_steps=20,
                    num_images_per_prompt=1,
                    image_guidance_scale=1.5 if op == 0 else random.uniform(1.01, 2.5)
                ).images[0])
        fig, ax = plt.subplots(1, n_ai_options, figsize=(2 * n_ai_options, 2))
        for j in range(n_ai_options):
            ax[j].imshow(target_imgs[j])
            ax[j].set_xticks([])
            ax[j].set_yticks([])
            ax[j].set_title(str(j))
        if n_ai_options > 1:
            plt.show()
            while (True):
                try:
                    target_img_ind = int(
                        input("输入您最喜欢的选项的编号？如果您不喜欢任何选项并希望更多选项，请输入 -1。\n:"))
                    break
                except:
                    print("那是一个数字吗？确保您输入的是您喜欢的图像的编号。如果您想生成替代品，请输入 -1。")
            if target_img_ind < 0:
                continue
            target_img = target_imgs[target_img_ind]
        else:
            target_img = target_imgs[0]
        satisfied = True
    target_img = torch.from_numpy(np.array(target_img)).permute(2, 0, 1).float().to(device)[None] / 255.
    return text_prompt, target_img


if __name__ == '__main__':
    opt = Options()
    opt.gather_options()

    cofrida_model = get_instruct_pix2pix_model(
        lora_weights_path=opt.cofrida_model,
        device=device)
    cofrida_model.set_progress_bar_config(disable=True)
    n_ai_options = 6

    date_and_time = datetime.datetime.now()
    run_name = '' + date_and_time.strftime("%m_%d__%H_%M_%S")
    opt.writer = TensorBoard('{}/{}'.format(opt.tensorboard_dir, run_name))
    opt.writer.add_text('args', str(sys.argv), 0)

    painter = Painter(opt)
    opt = painter.opt

    painter.to_neutral()
    # painter.move_robot_to_safe_position()

    w_render = int(opt.render_height * (opt.CANVAS_WIDTH_M / opt.CANVAS_HEIGHT_M))
    h_render = int(opt.render_height)
    opt.w_render, opt.h_render = w_render, h_render

    consecutive_paints = 0
    consecutive_strokes_no_clean = 0
    curr_color = -1

    color_palette = None
    if opt.use_colors_from is not None:
        color_palette = get_colors(cv2.resize(cv2.imread(opt.use_colors_from)[:, :, ::-1], (256, 256)),
                                   n_colors=opt.n_colors)
        opt.writer.add_image('paint_colors/using_colors_from_input', save_colors(color_palette), 0)

    for i in range(9):  # 最大回合数
        ##################################
        ########## 人类回合 ###########
        ##################################
        # painter.move_robot_to_safe_position()
        current_canvas = painter.camera.get_canvas_tensor() / 255.
        opt.writer.add_image('images/{}_0_canvas_start'.format(i), format_img(current_canvas), 0)
        current_canvas = Resize((h_render, w_render), antialias=True)(current_canvas)

        try:
            input('\n请随意绘画，然后完成后按回车。')
        except SyntaxError:
            pass

        current_canvas = painter.camera.get_canvas_tensor() / 255.
        opt.writer.add_image('images/{}_1_canvas_after_human'.format(i), format_img(current_canvas), 0)
        current_canvas = Resize((h_render, w_render), antialias=True)(current_canvas)

        #################################
        ########## 机器人回合 ###########
        #################################

        curr_canvas = painter.camera.get_canvas()
        curr_canvas_pil = Image.fromarray(curr_canvas.astype(np.uint8)).resize((512, 512))
        # save canvas
        curr_canvas_pil.save(r"D:\code\Frida-master\src\caches\sharpie_test\canvas.png")
        # 让用户生成并选择 cofrida 图像进行绘画
        if painter.opt.simulate:
            text_prompt, target_img = get_cofrida_image_to_draw(cofrida_model, curr_canvas_pil, n_ai_options)
        else:
            # 从本地路径读取图片
            text_prompt = """masterpiece, (best quality), highly detailed, ultra-detailed, white background, (minimal black), a Chinese mountain top, (ink painting style:1.2), misty atmosphere, (traditional brushstroke texture),
    serene, (high contrast), (flowing lines), subtle shading, intricate details, vast emptiness, (classic Chinese aesthetic), balance of light and shadow."""
            target_img = cv2.imread(r"D:\code\Frida-master\src\caches\sharpie_test\1.png")
            # 转换颜色通道，从 BGR 转为 RGB
            target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
            # 将图像数据转换为 float32 类型并归一化到 [0, 1] 范围
            target_img = target_img.astype(np.float32) / 255.0
            # 如果需要，调整图像的形状（例如，添加批次维度）
            target_img = torch.from_numpy(target_img).permute(2, 0, 1).float().to(device)[None]  # 添加批次维度

        opt.writer.add_image('images/{}_2_target_from_cofrida_{}'.format(i, text_prompt), format_img(target_img), 0)
        target_img = Resize((h_render, w_render), antialias=True)(target_img)
        # 询问使用多少笔画
        num_strokes = int(input("在这个计划中使用多少笔画？\n:"))

        # 生成初始（随机计划）
        painting = initialize_painting(opt, num_strokes, target_img,
                                       current_canvas.to(device), opt.ink, device=device)
        color_palette = None  # TODO: 支持输入固定调色板

        # 设置规划算法的变量
        opt.objective = ['clip_conv_loss']
        opt.objective_data_loaded = [target_img]
        opt.objective_weight = [1.0]

        # 获取计划
        painting, _ = optimize_painting(opt, painting,
                                        optim_iter=opt.optim_iter, color_palette=color_palette,
                                        log_title='{}_3_plan'.format(i))

        # 警告用户计划已准备好。准备绘画。
        if not painter.opt.simulate:
            show_img(painter.camera.get_canvas() / 255.,
                     title="初步计划完成。准备开始绘画。确保提供混合颜料，然后退出以开始绘画。")

        # 执行计划
        for stroke_ind in tqdm(range(len(painting)), desc="执行计划"):
            stroke = painting.pop()

            # 清洁画笔和/或获取更多颜料
            if not painter.opt.ink:
                color_ind, _ = nearest_color(stroke.color_transform.detach().cpu().numpy(),
                                             color_palette.detach().cpu().numpy())
                new_paint_color = color_ind != curr_color
                if new_paint_color or consecutive_strokes_no_clean > 12:
                    painter.clean_paint_brush()
                    painter.clean_paint_brush()
                    consecutive_strokes_no_clean = 0
                    curr_color = color_ind
                    new_paint_color = True
                if consecutive_paints >= opt.how_often_to_get_paint or new_paint_color:
                    painter.get_paint(color_ind)
                    consecutive_paints = 0
            elif consecutive_paints >= opt.how_often_to_get_paint:
                print("使用固定颜色的墨水，仅需根据蘸墨频率去获取颜料。")
                painter.get_paint(0)
                painter.rub_brush_on_rag()
                consecutive_paints = 0

            # 将画布比例坐标转换为机器人坐标
            x, y = stroke.transformation.xt.item() * 0.5 + 0.5, stroke.transformation.yt.item() * 0.5 + 0.5
            y = 1 - y
            x, y = min(max(x, 0.), 1.), min(max(y, 0.), 1.)  # 安全检查
            x_glob, y_glob, _ = canvas_to_global_coordinates(x, y, None, painter.opt)

            # 执行绘画
            stroke.execute(painter, x_glob, y_glob, stroke.transformation.a.item())
            consecutive_paints += 1
        # painter.move_robot_to_safe_position()
        # time.sleep(10)
        current_canvas = painter.camera.get_canvas_tensor() / 255.
        opt.writer.add_image('images/{}_4_canvas_after_drawing'.format(i), format_img(current_canvas), 0)
        current_canvas = Resize((h_render, w_render), antialias=True)(current_canvas)

    if not painter.opt.ink:
        painter.clean_paint_brush()
        painter.clean_paint_brush()

    painter.to_neutral()

    painter.robot.good_night_robot()