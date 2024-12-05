# Camera Intrinsic Calibration 
# Author: Jason Xu

import numpy as np
import cv2
from skimage import io

# 计算相机内参矩阵
# 代码来自卡内基梅隆大学15-463计算摄影课程

# images: 标定板图像文件名列表
# checkerboard: 棋盘格内角点的维度
# dW: 角点精化窗口大小。对于分辨率较低的图像应设置较小值
def computeIntrinsic(images, checkerboard, dW):    
    # 定义棋盘格的维度
    # 内角点数量，硬编码为课堂示例
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 80, 0.001)

    # 创建向量以存储每个棋盘格图像的3D点向量
    objpoints = []
    # 创建向量以存储每个棋盘格图像的2D点向量
    imgpoints = [] 

    # 定义3D点的世界坐标
    objp = np.zeros((1, checkerboard[0] * checkerboard[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:checkerboard[0], 0:checkerboard[1]].T.reshape(-1, 2)
    img_shape = None

    # 提取给定目录中单个图像的路径
    print('显示棋盘格角点。按任意键继续到下一个示例')
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img_shape = gray.shape[::-1]
        
        # 查找棋盘格角点
        # 如果在图像中找到所需的角点，则ret为True
        ret, corners = cv2.findChessboardCorners(gray, checkerboard, None)
        
        """
        如果检测到所需数量的角点，
        我们将细化像素坐标并将其显示在棋盘格图像上
        """
        if ret == True:
            objpoints.append(objp)
            # 细化给定2D点的像素坐标
            # print(corners)
            corners2 = cv2.cornerSubPix(gray, corners, dW, (-1,-1), criteria)
            
            imgpoints.append(corners2)

            # 绘制并显示角点
            img = cv2.drawChessboardCorners(img, checkerboard, corners2, ret)
        else:
            print("错误：未找到棋盘格")
        
        io.imshow(img)
        io.show()

    cv2.destroyAllWindows()

    """
    通过传递已知的3D点（objpoints）
    和检测到的角点对应的像素坐标（imgpoints）
    执行相机标定
        ret:
        mtx: 相机矩阵
        dist: 畸变系数
    """
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_shape, None, None)

    print("相机矩阵: \n")
    print(mtx)
    print("畸变系数: \n")
    print(dist)

    newCameraMtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, img_shape, 0, img_shape)

    return mtx, dist, newCameraMtx, roi