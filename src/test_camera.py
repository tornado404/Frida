from torch import nn

from camera.dslr import WebCam
from options import Options
from paint_utils3 import show_img, random_init_painting

if __name__ == "__main__":
    import json

    opt = Options()
    opt.gather_options()


    # 读取 JSON 文件
    from src.brush_stroke import BrushStroke

    with open('painting_data.json', 'r', encoding='utf-8') as f:  # 读取文件
        import json

        data = json.load(f)
        strokes_array = []
        for stroke_data in data['painting']['brush_strokes']:  # 遍历反序列化 brush_strokes
            strokes_array.append(BrushStroke.from_dict(stroke_data, opt))


    # 打印长度
    print(f"rush_strokes 数组的长度: {len(strokes_array)}")

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
    
