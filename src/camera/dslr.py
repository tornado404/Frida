#! /usr/bin/env python3
##########################################################
#################### Copyright 2022 ######################
################ by Peter Schaldenbrand ##################
### The Robotics Institute, Carnegie Mellon University ###
################ All rights reserved. ####################
##########################################################

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from .dslr_gphoto import camera_init

matplotlib.rcParams['font.family'] = 'SimHei'  # 使用黑体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号 '-' 显示为方块的问题

import pickle
import os

import torch 
from torchvision.transforms import Resize

# import camera.color_calib
from camera.color_calib import color_calib, find_calib_params
from camera.harris import find_corners
from camera.intrinsic_calib import computeIntrinsic
import glob

from .dslr_gphoto import *

# https://github.com/IntelRealSense/librealsense/blob/master/wrappers/python/examples/opencv_viewer_example.py

class WebCam():
    def __init__(self, opt, debug=False):
        # 初始化相机
        self.camera = camera_init()
        self.camera.release()
        # 调试模式
        self.debug = debug
        # 画布变换矩阵
        self.H_canvas = None

        # 是否有颜色信息
        self.has_color_info = False

        # 颜色变换矩阵和灰度值
        self.color_tmat = None
        self.greyval = None

        # 选项配置
        self.opt = opt


    def get_rgb_image(self, channels='rgb'):
        # 获取RGB图像
        self.camera = camera_init()
        target, img = capture_image(self.camera)
        self.camera.release()

        if channels == 'rgb':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为RGB格式
        elif channels == 'bgr':
            pass  # 保持BGR格式
        else:
            raise ValueError("Unsupported channel format: {}".format(channels))

        return target, img

    # 返回RGB图像，经过颜色校正
    def get_color_correct_image(self, use_cache=False):
        print("Getting color-corrected image")
        if not self.has_color_info and self.opt.calib_colors:
            if not use_cache or not os.path.exists(os.path.join(self.opt.cache_dir, 'cached_color_calibration.pkl')):
                try:
                    input('未找到颜色信息。开始颜色校准。请确保已将Macbeth颜色检查器放置在相机视野内并按ENTER继续。')
                except SyntaxError:
                    pass
                completed_color_calib = False
                while not completed_color_calib:
                    try:
                        self.init_color_calib()
                        retake = input("重新拍摄？y/[n]")
                        if not(retake[:1] == 'y' or retake[:1] == 'Y'):
                            completed_color_calib = True
                    except Exception as e:
                        print(e)
                        try: input('无法校准。移动颜色检查器并重试（按回车键准备）')
                        except SyntaxError: pass 
                try:    
                    input("移除颜色检查器，然后按回车键。")
                except SyntaxError:
                    pass
            else:
                params = pickle.load(open(os.path.join(self.opt.cache_dir, "cached_color_calibration.pkl"),'rb'), encoding='latin1')
                self.color_tmat, self.greyval = params["color_tmat"], params["greyval"]
                self.has_color_info = True


        path, img = self.get_rgb_image()

        # 必须这样做
        if self.opt.calib_colors:
            return cv2.cvtColor(color_calib(img, self.color_tmat, self.greyval), cv2.COLOR_BGR2RGB)
        else:
            return img

    def get_canvas(self, use_cache=False, max_height=None):
        if self.H_canvas is None:
            print("H_canvas is None, calibrating canvas")
            self.calibrate_canvas(use_cache)
        
        # 如果可能，使用校正后的图像
        if (self.has_color_info and self.opt.calib_colors):
            print("Using color-corrected image")
            img = self.get_color_correct_image(use_cache)
        else:
            print("Using raw image")
            _, img = self.get_rgb_image()

        canvas = cv2.warpPerspective(img, self.H_canvas, (img.shape[1], img.shape[0]))
        w = int(img.shape[0] * (self.opt.CANVAS_WIDTH_M/self.opt.CANVAS_HEIGHT_M))
        canvas = canvas[:, :w]
        if max_height is not None and canvas.shape[0] > max_height:
            fact = 1.0 * img.shape[0] / max_height
            canvas = cv2.resize(canvas, (int(canvas.shape[1]/fact), int(canvas.shape[0]/fact)))
        return canvas
    
    def get_canvas_tensor(self, h=None, w=None):
        canvas = self.get_canvas()
        canvas = torch.from_numpy(canvas).permute(2,0,1).unsqueeze(0)
        if h is not None and w is not None:
            canvas = Resize((h,w), antialias=True)(canvas)
        canvas = torch.cat([canvas, torch.ones(1,1,canvas.shape[2],canvas.shape[3])], dim=1)
        cv2.imwrite(os.path.join(os.getcwd(), "canvas.jpg"), canvas.squeeze(0).permute(1, 2, 0).numpy() * 255)  # 保存为JPEG格式
        return canvas

    def calibrate_canvas(self, use_cache=False):
        img = self.get_color_correct_image(use_cache=use_cache)
        h = img.shape[0]
        print(f"cached img shape: {img.shape}")
        # 原始图像的宽高比与纸张相比过宽
        # w = int(h * LETTER_WH_RATIO)
        w = int(h * (self.opt.CANVAS_WIDTH_M/self.opt.CANVAS_HEIGHT_M))

        print(f"计算的宽度: {w}, 图像宽度: {img.shape[1]}")  # 添加调试信息
        assert(w <= img.shape[1])

        if use_cache and os.path.exists(os.path.join(self.opt.cache_dir, 'cached_H_canvas.pkl')):
            self.H_canvas = pickle.load(open(os.path.join(self.opt.cache_dir, "cached_H_canvas.pkl"),'rb'), encoding='latin1')
            img1_warp = cv2.warpPerspective(img, self.H_canvas, (img.shape[1], img.shape[0]))
            # plt.imshow(img1_warp[:, :w])
            # plt.title('Hopefully this looks like just the canvas')
            # plt.show()
            return

        self.canvas_points = find_corners(img, show_search=self.debug)

        img_corners = img.copy()
        for corner_num in range(4):
            x, y = self.canvas_points[corner_num]

            # 反转颜色以显示
            for u in range(-10, 10):
                for v in range(-10, 10):
                    img_corners[y+u, x+v, :] = np.array((255, 255, 255)) - img_corners[y+u, x+v, :]

        plt.clf()
        plt.imshow(img_corners)
        plt.title("Here are the found corners")
        plt.show()

        true_points = np.array([[0,0],[w,0], [w,h],[0,h]])
        self.H_canvas, _ = cv2.findHomography(self.canvas_points, true_points)
        img1_warp = cv2.warpPerspective(img, self.H_canvas, (img.shape[1], img.shape[0]))
        
        # print(img1_warp[:, :w].shape)
        # print(img1_warp.shape)
        plt.imshow(img1_warp[:, :w])
        plt.title('Hopefully this looks like just the canvas')
        plt.show()
        # plt.imshow(img1_warp)
        # plt.title('Hopefully this looks like just the canvas')
        # plt.show()
        
        with open(os.path.join(self.opt.cache_dir, 'cached_H_canvas.pkl'),'wb') as f:
            pickle.dump(self.H_canvas, f)

    def init_color_calib(self):
        path, img = self.get_rgb_image()
        self.color_tmat, self.greyval = find_calib_params(path, self.debug)
        self.has_color_info = True
        
        with open(os.path.join(self.opt.cache_dir, 'cached_color_calibration.pkl'),'wb') as f:
            params = {"color_tmat":self.color_tmat, "greyval":self.greyval}
            pickle.dump(params, f)

    # 相机的内参标定
    def init_distortion_calib(self, imgs_exist=False, calib_path='./calibration/', num_imgs=10):
        # 如果图像不存在，则捕获图像
        if not imgs_exist:
            # 捕获设定数量的图像，i可编辑以启用重拍
            i = 0
            while i < num_imgs:
                input("移动棋盘格并按ENTER捕获图像 %d/%d." % ((i + 1), num_imgs))
                _, img = self.get_rgb_image()
                plt.imshow(img)
                plt.draw()
                plt.show(block=False)
                plt.pause(0.01)
                # 如果需要重拍
                retake = input("重拍？y/[n]")
                plt.close()
                if retake[:1] == 'y' or retake[:1] == 'Y':
                    # 不保存并且不递增
                    print("重拍。")
                    continue
                else:
                    fname = calib_path + str(i).zfill(3) + ".jpg"
                    plt.imsave(fname, img)
                    print("保存到 " + fname + ".")
                    i += 1

        images = glob.glob(calib_path + "*.jpg")
        self.intrinsics = computeIntrinsic(images, (6, 8), (8, 8))
    
    # 使用OpenCV进行去畸变和裁剪
    # 来自OpenCV教程
    def undistort(self, img):
        if self.intrinsics is None:
            input("未找到内参矩阵。您必须执行内参标定。")
            quit()
        # 去畸变
        mtx, dist, newCameraMtx, roi = self.intrinsics
        dst = cv2.undistort(img, mtx, dist, None, newCameraMtx)
        # 裁剪图像
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
        return dst

class SimulatedWebCam():
    def __init__(self, opt):
        self.opt = opt
        w_h_ratio = float(opt.CANVAS_WIDTH_M) / opt.CANVAS_HEIGHT_M
        self.h = 1024
        self.w = int(self.h * w_h_ratio)
        h = self.h
        self.canvas = np.ones((h,int(h * w_h_ratio),3), dtype=np.float32) * 255.
    def get_canvas(self):
        return self.canvas
    def get_canvas_tensor(self, h=None, w=None):
        canvas = self.get_canvas()
        canvas = torch.from_numpy(canvas).permute(2,0,1).unsqueeze(0)
        if h is not None and w is not None:
            canvas = Resize((h,w), antialias=True)(canvas)
        else:
            h,w = self.h, self.w
        canvas = torch.cat([canvas, torch.ones(1,1,h,w)], dim=1)
        return canvas
    def calibrate_canvas(self, use_cache=False):
        pass