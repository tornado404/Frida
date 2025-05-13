import datetime
import json
import os
from typing import Optional

import torchvision
from loguru import logger
import threading

import torch
from torch import nn

from src.brush_stroke import BrushStroke
from src.my_tensorboard import TensorBoard

from src.options import Options
from src.paint_utils3 import random_init_painting, nearest_color, canvas_to_global_coordinates
from src.painter import Painter
from src.painting_optimization import load_objectives_data, optimize_painting

class TaskManager:
    def __init__(self):
        self.tasks = {}
        self.run_thread = None
        self.painter = None


    def start_task(self, task_id, painter, opt, options):
        if task_id in self.tasks and self.tasks[task_id]['status'] == 'running':
            return "Task already running"
        self.painter = painter

        self.tasks[task_id] = {
            'status': 'running',
            'result': None,
            'is_painting': True,      # 是否正在绘画中
            'should_stop': False,     # 是否应该停止绘画
            'total_strokes': 0,       # 总笔画数
            'completed_strokes': 0,   # 已完成的笔画数
            'current_color_index': 0, # 当前正在绘制的颜色索引
            'total_colors': 0,        # 总颜色数
            'error': None            # 错误信息
        }
        
        def painting_thread():
            try:
                result = start_painting(self.painter, opt, options)
                logger.debug("painting_thread: start_painting completed")
                self.tasks[task_id]['result'] = result
                self.tasks[task_id]['is_painting'] = False
                self.tasks[task_id]['status'] = 'completed'
            except Exception as e:
                logger.error(f"Painting error: {str(e)}")
                self.tasks[task_id]['error'] = str(e)
                self.tasks[task_id]['status'] = 'error'

        self.run_thread = threading.Thread(target=painting_thread)
        self.run_thread.daemon = True
        self.run_thread.start()
        
        
        return "Task started successfully"

    def stop_task(self, task_id):
        if task_id in self.tasks and self.tasks[task_id]['status'] == 'running':
            logger.debug("stop task {}", task_id)
            self.tasks[task_id]['should_stop'] = True
            self.tasks[task_id]['status'] = 'stopped'
            if self.run_thread and self.run_thread.is_alive():
                self.run_thread.join(timeout=5.0)  # 等待线程结束，最多等待5秒
            return True
        return False

    def get_task_status(self, task_id):
        """获取任务状态"""
        if task_id in self.tasks:
            return self.tasks[task_id]
        return {'error': 'task not_found'}

    def get_latest_canvas_image(self):
        if self.painter is not None:
            return self.painter.get_latest_canvas()
        return None

manager = TaskManager()

opt = None

# 获取任务状态，如果任务不存在则使用默认状态

# 删除全局painting_status变量，改为在函数内部获取task_status

def init_painter(options):
    """初始化绘画机器人"""
    # 初始化配置
    opt = Options()
    opt.gather_options()
    # 设置运行名称和tensorboard
    date_and_time = datetime.datetime.now()
    run_name = '' + date_and_time.strftime("%m_%d__%H_%M_%S")
    opt.writer = TensorBoard('{}/{}'.format(opt.tensorboard_dir, run_name))
    import sys
    sysargv = str(sys.argv)
    logger.debug("sysargv is {}", sysargv)
    opt.writer.add_text('args', sysargv, 0)
    # 递归更新配置
    def update_dict(obj, update_dict):
        """
        递归更新字典配置
        Args:
            obj: 需要被更新的成员变量
            update_dict: 用于更新的字典数据
        Returns:
            更新后的字典
        """
        for key, value in update_dict.items():

            if isinstance(value, dict):
                if hasattr(obj, key):
                    update_dict(getattr(obj, key), value)
            else:
                # obj不存在该属性key，需要加上该属性
                setattr(obj, key, value)
        return obj

    # 将options转换为字典
    update_dict(opt, options)
    # 初始化画家
    testopt = opt
    logger.debug("ink:{}", opt.ink)
    logger.debug("init_optim_iter:{}", opt.init_optim_iter)
    logger.debug("lr_multiplier:{}", opt.lr_multiplier)
    logger.debug("num_strokes:{}", opt.num_strokes)
    painter = Painter(opt)
    opt = painter.opt

    return painter, opt


def start_painting(painter, opt, options) -> str:
    """开始绘画流程"""
    # 重置绘画状态
    # 获取任务状态的引用
    task_status = manager.get_task_status('painting_task')
    # 更新状态
    task_status.update({
        'is_painting': True,
        'should_stop': False,
        'total_strokes': 0,
        'completed_strokes': 0,
        'current_color_index': 0,
        'total_colors': 0,
        'error': None
    })

    # # 设置渲染尺寸
    w_render = int(opt.render_height * (opt.CANVAS_WIDTH_M/opt.CANVAS_HEIGHT_M))
    h_render = int(opt.render_height)
    opt.w_render, opt.h_render = w_render, h_render

    consecutive_paints = 0
    consecutive_strokes_no_clean = 0
    curr_color = -1
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    color_palette = torch.tensor(opt.color_palette, dtype=torch.float32).to(device)

    # 获取当前画布
    # 获取初始画布
    current_canvas = manager.painter.camera.get_canvas_tensor(h=h_render,w=w_render).to(device) / 255.
    # 保存初始画布到本地磁盘
    current_dir = os.path.dirname(os.path.abspath(__file__))
    canvas_save_path = os.path.join(current_dir, 'initial_canvas.png')
    torchvision.utils.save_image(current_canvas, canvas_save_path)
    x1, x2, y1, y2 = manager.painter.opt.X_CANVAS_MIN, manager.painter.opt.X_CANVAS_MAX, manager.painter.opt.Y_CANVAS_MIN, manager.painter.opt.Y_CANVAS_MAX
    logger.info("CANVAS bound: X [{}, {}] Y[{}, {}]", x1, x2, y1, y2)
    manager.painter.start_recording()

    # 加载目标数据
    load_objectives_data(opt)

    # 初始化绘画
    painting = random_init_painting(opt, current_canvas, opt.num_strokes, ink=opt.ink)
    painting.to(device)
    # 从缓存加载或优化绘画
    if not os.path.exists('painting_data.json'):
        painting, color_palette = optimize_painting(opt, painting,
                                                    optim_iter=opt.init_optim_iter, color_palette=color_palette)
        import json
        with open('painting_data.json', 'w', encoding='utf-8') as f:
            json.dump({
                'painting': {
                    'brush_strokes': [painting.pop().to_dict() for _ in range(len(painting.brush_strokes))],
                },
            }, f, ensure_ascii=False)

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

    # 执行绘画
    strokes_per_adaptation = int(len(painting) / opt.num_adaptations)
    strokes_array = []
    total_strokes = min(len(painting), strokes_per_adaptation)
    task_status['total_strokes'] = total_strokes

    for stroke_ind in range(total_strokes):
        stroke = painting.pop()
        strokes_array.append(stroke)

    # 按颜色执行绘画
    temp_strokes = strokes_array.copy()
    color_palette_rgb = [[int(c[0]), int(c[1]), int(c[2])] for c in color_palette.cpu().numpy()]  # 将颜色转换为RGB格式
    color_index_list = []
    task_status['total_colors'] = len(color_palette)
    for stroke_ind, stroke in enumerate(temp_strokes):
        color_ind, _ = nearest_color(stroke.color_transform.detach().cpu().numpy(),
                                     color_palette.detach().cpu().numpy())
        color_index_list.append(color_ind)


    for index, color in enumerate(color_palette):
        task_status['current_color_index'] = index
        temp_strokes = strokes_array.copy()
        consecutive_paints = 0
        for stroke_ind, stroke in enumerate(temp_strokes):
            if task_status['should_stop']:
                task_status['is_painting'] = False
                task_status['error'] = "用户手动终止绘画"
                return "绘画已终止"
            if color_index_list[stroke_ind] != index:
                continue

            if consecutive_paints % opt.how_often_to_get_paint == 0:
                manager.painter.get_paint(index)
                manager.painter.rub_brush_on_rag()
                consecutive_paints = 0

            # 转换坐标
            x, y = stroke.transformation.xt.item() * 0.5 + 0.5, stroke.transformation.yt.item() * 0.5 + 0.5
            y = 1 - y
            x, y = min(max(x, 0.), 1.), min(max(y, 0.), 1.)  # safety
            x_glob, y_glob, _ = canvas_to_global_coordinates(x, y, None, manager.painter.opt)

            # 执行绘画
            stroke.execute(manager.painter, x_glob, y_glob, stroke.transformation.a.item())
            consecutive_paints += 1
            task_status['completed_strokes'] += 1

        manager.painter.clean_paint_brush()
        manager.painter.clean_paint_brush()

        if index != len(color_palette) - 1:
            manager.painter.get_paint(index + 1)
        manager.painter.rub_brush_on_rag()
    manager.painter.stop_recording()

    return "绘画完成！"
