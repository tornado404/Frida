import os
import cv2
import numpy as np
from loguru import logger

def canvas_record(cnp):
    """
    cnp: [((x, y, z), ((x, y, z), ...]
    """
    touch_depth = 100 # 贴近纸面时的高度，代表基础高度，
    press_depth = 5 # 按压时的最大高度，代表毛笔从贴近纸面到按压时的最大深度
    height, width = 1080, 1920
    video_dir = os.path.join(os.getcwd(), "canvas")
    video_file = os.path.join(video_dir, "simplified.mp4")
    output_dir = os.path.dirname(video_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_file, fourcc, 30, (int(width), int(height)))
    if not out.isOpened():
        logger.error("Failed to open video writer")
        return
    # 白色画布
    canvas = np.full((int(height), int(width), 3), 255, dtype=np.uint8)
    prev_point = None
    
    # 预先遍历一遍，将z值归一化到0-1
    max_z = max(cnp, key=lambda x: x[2])[2]
    min_z = min(cnp, key=lambda x: x[2])[2]
    logger.debug(f"max_z {max_z} min_z {min_z}")
    #  遍历过程同时计算连续的前后两点在 xoy平面的平均间距
    avg_dis = np.mean([np.sqrt((cnp[i][0] - cnp[i - 1][0]) ** 2 + (cnp[i][1] - cnp[i - 1][1]) ** 2) for i in range(1, len(cnp))])
    logger.debug(f"avg_dis {avg_dis}")
    
    for i, (x, y, z) in enumerate(cnp):
        cnp[i] = (x, y, (z - min_z) / (max_z - min_z))

    for i in range(len(cnp) - 1):
        (x1, y1, z1) = cnp[i]
        (x2, y2, z2) = cnp[i + 1]
        # avg_dis vs dis 比较，当dis小于avg_dis时，插入10个中间点，否则不插入
        dis = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if dis < avg_dis:
            steps = 1  # 插入10个中间点
            for step in range(steps + 1):
                t = step / steps
                x = int(x1 + (x2 - x1) * t)
                y = int(y1 + (y2 - y1) * t)
                z = z1 + (z2 - z1) * t
                center = (x, y)
                wid = int(z * press_depth)
                logger.debug(f"Frame {i} draw_on_video x {x} y {y}, z {z} wid {wid}")
                if prev_point:
                    cv2.line(canvas, prev_point, center, (0, 0, 0), wid if wid > 0 else 1)
                else:
                    cv2.circle(canvas, center, 3, (0, 255, 0), -1)
                prev_point = center
                out.write(canvas)
        else:
            #
            center = (int(x), int(y))
            wid = int(z * press_depth)
            logger.debug(f"Frame {i} draw_on_video x {x} y {y}, z {z} wid {wid}")
            cv2.circle(canvas, center, 3, (0, 255, 0), -1)

            prev_point = center
            out.write(canvas)
        
    out.release()